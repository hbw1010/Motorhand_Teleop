#include "transport.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using real_hand::Angles;
using real_hand::Feedback;
using real_hand::monotonic_ns;
using Clock = std::chrono::steady_clock;
struct LatchedFault : std::runtime_error {
    using std::runtime_error::runtime_error;
};
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void message(char *buffer, size_t size, const char *text) noexcept {
    if (buffer && size) std::snprintf(buffer, size, "%s", text);
}
void positive(double value, const char *name) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(name);
}
void append(std::string &errors, const std::exception &e) {
    if (!errors.empty()) errors += "; ";
    errors += e.what();
}
}

struct RealHand {
    RealHandOptions options;
    std::string serial, address, approved, preflight;
    std::unique_ptr<real_hand::Transport> transport;
    real_hand::Identity identity;
    std::mutex control;
    std::atomic<bool> terminate{false}, interrupt{false};
    std::thread watchdog;
    int phase = 0;
    bool connected = false, attempted = false, stopped_confirmed = false;
    bool authorized = false;
    std::string fault;
    bool fault_pending = false;
    Angles position{}, velocity{};
    uint64_t step_at = 0, command_at = 0, source_deadline_ns = 0;

    explicit RealHand(const RealHandOptions &o) : options(o) {
        if (o.model != 2) throw std::invalid_argument("本轮仅支持 model=2 的 Wuji Hand2");
        if (o.hand != 0 && o.hand != 1) throw std::invalid_argument("hand 必须为 0 或 1");
        if (!o.serial_number || !*o.serial_number) throw std::invalid_argument("必须显式提供序列号");
        serial = o.serial_number;
        if (static_cast<unsigned char>(serial.front()) <= ' ' || static_cast<unsigned char>(serial.back()) <= ' ')
            throw std::invalid_argument("序列号不能包含首尾空白");
        address = o.address ? o.address : "";
        approved = o.approved_firmware_version ? o.approved_firmware_version : "";
        preflight = o.preflight_firmware_version ? o.preflight_firmware_version : "";
        if (!std::isfinite(o.max_velocity) || o.max_velocity < 0 ||
            !std::isfinite(o.max_acceleration) || o.max_acceleration < 0 ||
            (o.max_velocity == 0) != (o.max_acceleration == 0) ||
            !std::isfinite(o.realtime_max_velocity) || o.realtime_max_velocity < 0)
            throw std::invalid_argument("可选主机速度必须非负有限，回零速度和加速度须同时配置");
        positive(o.command_timeout, "command_timeout 必须有限且为正");
        positive(o.feedback_timeout, "feedback_timeout 必须有限且为正");
        positive(o.home_rate_hz, "home_rate_hz 必须有限且为正");
        positive(o.home_tolerance, "home_tolerance 必须有限且为正");
        positive(o.home_timeout, "home_timeout 必须有限且为正");
        positive(o.enable_timeout, "enable_timeout 必须有限且为正");
        positive(o.transition_duration, "transition_duration 必须有限且为正");
        if (o.home_rate_hz > 1000 || o.timeout_ms == 0 || o.auto_home > 1 ||
            !std::isfinite(o.effort_limit) || (o.effort_limit < 0 && o.effort_limit != -1) ||
            o.effort_limit > std::numeric_limits<float>::max() ||
            !std::isfinite(o.mit_kp) || (o.mit_kp < 0 && o.mit_kp != -1) || o.mit_kp > 10 ||
            !std::isfinite(o.mit_kd) || (o.mit_kd < 0 && o.mit_kd != -1) || o.mit_kd > 1 ||
            (o.mit_kp == -1) != (o.mit_kd == -1))
            throw std::invalid_argument("控制参数越界：可省略设备参数，否则力限非负、kp<=10、kd<=1");
        options.serial_number = serial.c_str();
        options.address = address.c_str();
        options.approved_firmware_version = approved.c_str();
        options.preflight_firmware_version = preflight.c_str();
        transport = real_hand::make_transport();
        watchdog = std::thread([this] { watch(); });
    }
    ~RealHand() {
        terminate.store(true);
        interrupt.store(true);
        if (watchdog.joinable()) watchdog.join();
        try {
            std::lock_guard<std::mutex> lock(control);
            if (connected) shutdown(false, true);
        } catch (...) {}
    }
    void require_ready() const {
        if (!fault.empty()) throw LatchedFault(fault);
        require(connected && phase != 4, "必须先显式 connect，关闭后不得隐式重连");
    }
    void validate(const Angles &values) const {
        for (size_t i = 0; i < 20; ++i)
            if (!std::isfinite(values[i]) || values[i] < identity.lower[i] || values[i] > identity.upper[i])
                throw std::invalid_argument("角度必须为限位内的有限弧度，禁止静默裁剪");
    }
    Feedback sample(bool active, bool allow_error = false) {
        auto f = transport->feedback();
        require(f.complete && f.position_ns && f.diagnostics_ns, "缺少完整20关节位置、在线或使能反馈");
        const uint64_t now = monotonic_ns();
        require(f.position_ns <= now && f.diagnostics_ns <= now, "反馈主机时间无效");
        require((now - std::min(f.position_ns, f.diagnostics_ns)) * 1e-9 <= options.feedback_timeout,
                "位置或诊断反馈超时");
        for (size_t i = 0; i < 20; ++i) {
            require(f.online[i] == 1 && f.enabled[i] <= 1, "关节离线或使能状态未知");
            require(std::isfinite(f.positions[i]), "位置反馈含非有限值");
            if (!allow_error) require(f.errors[i] == 0 || f.warnings[i], "关节报告停机故障或未知错误码");
            if (active) require(f.enabled[i] == 1, "控制过程中关节失去使能");
        }
        return f;
    }
    void connect() {
        if (!fault.empty()) throw LatchedFault(fault);
        require(phase != 4, "已关闭实例不可重连");
        if (connected) return;
        try {
            identity = transport->connect(options);
            connected = true;
            require(identity.model == options.model && identity.hand == options.hand && identity.serial == serial,
                    "设备型号、手侧或序列号不匹配");
            require(identity.firmware.size() < 128, "固件版本长度超过ABI容量");
            for (size_t i = 0; i < 20; ++i)
                require(std::isfinite(identity.lower[i]) && std::isfinite(identity.upper[i]) &&
                        identity.lower[i] < identity.upper[i] && identity.lower[i] <= 0 && identity.upper[i] >= 0,
                        "关节限位无效或不包含零位");
            const auto deadline = Clock::now() + std::chrono::duration<double>(options.feedback_timeout);
            while (!transport->feedback().complete && Clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            sample(false, true);
            phase = 1;
        } catch (const std::exception &e) {
            fault = e.what();
            fault_pending = true;
            try { transport->release(); } catch (const std::exception &cleanup) { append(fault, cleanup); }
            connected = false;
            phase = 3;
            throw LatchedFault(fault);
        }
    }
    void wait_enabled(bool expected, uint64_t after) {
        const auto deadline = Clock::now() + std::chrono::duration<double>(options.enable_timeout);
        do {
            auto f = sample(false, !expected);
            if (f.diagnostics_ns > after && std::all_of(f.enabled.begin(), f.enabled.end(),
                    [expected](uint8_t v) { return v == static_cast<uint8_t>(expected); })) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (Clock::now() < deadline);
        throw std::runtime_error(expected ? "未从新鲜反馈确认全部关节使能" : "未从新鲜反馈确认停止，须人工确认安全");
    }
    Feedback wait_cleared(uint64_t after) {
        const auto deadline = Clock::now() + std::chrono::duration<double>(options.enable_timeout);
        // 2.13.1实机：清错RPC返回后1–2ms的新鲜Ready帧仍可早于设备完成清错，
        // 此时立即enable会被部分关节忽略。连续观察100ms，再取窗口后的新鲜帧。
        // 这是启动阶段的保守间隔，不是设备完成回执；使能后仍须逐轴确认。
        constexpr uint64_t settle_ns = 100'000'000;
        uint64_t healthy_since = 0;
        do {
            require(!interrupt.load() && !terminate.load(), "清错等待被显式停止请求中断");
            auto f = sample(false, true);
            require(std::all_of(f.enabled.begin(), f.enabled.end(), [](uint8_t v) { return v == 0; }),
                    "清错过程中关节意外使能");
            bool cleared = true;
            for (size_t i = 0; i < 20; ++i)
                cleared = cleared && (f.errors[i] == 0 || f.warnings[i]);
            if (f.position_ns > after && f.diagnostics_ns > after && cleared) {
                if (!healthy_since) healthy_since = monotonic_ns();
                if (std::min(f.position_ns, f.diagnostics_ns) >= healthy_since + settle_ns) return f;
            } else {
                healthy_since = 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (Clock::now() < deadline);
        throw std::runtime_error("未从清错后的新鲜位置及诊断确认全部关节失能且无停机故障或未知错误码");
    }
    uint64_t step_home(const Angles &target) {
        const uint64_t now = monotonic_ns();
        const double elapsed = (now - step_at) * 1e-9;
        const double dt = std::min(elapsed, 1.0 / options.home_rate_hz);
        if (dt <= 0) return 0;
        const double a = options.max_acceleration;
        Angles next{}, next_velocity{};
        for (size_t i = 0; i < 20; ++i) {
            const double delta = target[i] - position[i];
            // 离散制动距离包含本周期位移，不在抵达目标时突变速度。
            auto brake = [a, dt](double distance) {
                return std::sqrt(a * a * dt * dt + 2 * a * std::max(0.0, distance)) - a * dt;
            };
            const double desired = std::copysign(std::min(options.max_velocity, brake(std::abs(delta))), delta);
            const double lo = std::max(-options.max_velocity, velocity[i] - a * dt);
            const double hi = std::min(options.max_velocity, velocity[i] + a * dt);
            // home保证下一步间隔至少一个完整周期；落点速度须能在该周期受限降至0。
            const double arrival_velocity = delta / dt;
            const bool arriving = arrival_velocity >= lo && arrival_velocity <= hi &&
                                  std::abs(arrival_velocity) <= a * dt;
            const double v = arriving ? arrival_velocity : std::clamp(desired, lo, hi);
            const double safe_lo = -brake(position[i] - identity.lower[i]);
            const double safe_hi = brake(identity.upper[i] - position[i]);
            // 精确落点由validate检查限位；其制动已由上面的下一周期可停止条件保证。
            require(arriving || (v >= safe_lo - 1e-8 && v <= safe_hi + 1e-8),
                    "运动状态已无法在关节限位内安全制动");
            next[i] = arriving ? target[i] : position[i] + v * dt;
            next_velocity[i] = v;
        }
        validate(next);
        transport->send(next);
        position = next;
        velocity = next_velocity;
        step_at = monotonic_ns();
        return step_at;
    }
    void track(const Angles &target, uint64_t deadline_ns) {
        const uint64_t now = monotonic_ns();
        Angles next = target;
        if (options.realtime_max_velocity > 0) {
            const double max_step = options.realtime_max_velocity * (now - step_at) * 1e-9;
            for (size_t i = 0; i < 20; ++i)
                next[i] = position[i] + std::clamp(target[i] - position[i], -max_step, max_step);
        }
        validate(next);
        require(monotonic_ns() < deadline_ns, "原始命令已过期，禁止发送");
        transport->send(next);
        position = next;
        // 下一帧只能使用本帧发布完成之后的时间，不能把发送阻塞计入运动预算。
        step_at = monotonic_ns();
    }
    void home() {
        require(!interrupt.load() && !terminate.load(), "回零被显式停止请求中断");
        auto f = sample(true);
        validate(f.positions);
        position = f.positions;
        velocity.fill(0);
        step_at = monotonic_ns();
        const Angles start = position;
        const uint64_t transition_at = step_at;
        const Angles zero{};
        validate(zero);
        const auto deadline = Clock::now() + std::chrono::duration<double>(options.home_timeout);
        const auto period = std::chrono::duration<double>(1.0 / options.home_rate_hz);
        uint64_t sent_at = 0;
        do {
            require(!interrupt.load() && !terminate.load(), "回零被显式停止请求中断");
            f = sample(true);
            require(!interrupt.load() && !terminate.load(), "回零被显式停止请求中断");
            if (Clock::now() >= deadline) break;
            bool settled = sent_at && f.position_ns > sent_at;
            for (size_t i = 0; i < 20; ++i)
                settled = settled && std::abs(f.positions[i]) <= options.home_tolerance && velocity[i] == 0;
            if (settled) {
                position = zero;
                step_at = monotonic_ns();
                return;
            }
            const double elapsed = (monotonic_ns() - transition_at) * 1e-9;
            const double u = std::clamp(elapsed / options.transition_duration, 0.0, 1.0);
            const bool final_reference = u == 1.0;
            const double weight = std::clamp(u * u * u * (10 + u * (-15 + 6 * u)), 0.0, 1.0);
            Angles reference{};
            if (!final_reference)
                for (size_t i = 0; i < 20; ++i) reference[i] = start[i] * (1 - weight);
            if (options.max_velocity == 0) {
                // 仅回零使用固定起点的渐变参考；正常track路径保持原样。
                validate(reference);
                transport->send(reference);
                position = reference;
                // 重发不能推迟新鲜度基准，否则滞后的有效反馈会永远无法验收。
                if (final_reference && !sent_at) sent_at = monotonic_ns();
            } else {
                const auto published_at = step_home(reference);
                // 显式约束可能延后实际到零；首次最终零发布即固定反馈基准。
                // settled另行等待速度归零，不能以重发零目标推迟基准。
                if (final_reference && !sent_at && published_at &&
                    std::all_of(position.begin(), position.end(), [](double v) { return v == 0; }))
                    sent_at = published_at;
            }
            // 保持发送周期；等待可被显式停止打断，且绝不睡过回零deadline。
            const auto next_send = std::min(Clock::now() + period, deadline);
            while (Clock::now() < next_send) {
                require(!interrupt.load() && !terminate.load(), "回零被显式停止请求中断");
                const auto wake = std::min(Clock::now() + std::chrono::duration<double>(.002), next_send);
                std::this_thread::sleep_until(wake);
            }
        } while (Clock::now() < deadline);
        char detail[160];
        std::snprintf(detail, sizeof(detail), "回零超时：home_tolerance=%.9g rad，timeout=%.9g s",
                      options.home_tolerance, options.home_timeout);
        std::string error = detail;
        constexpr const char *fingers[] = {"thumb", "index", "middle", "ring", "pinky"};
        bool outside_tolerance = false;
        for (size_t i = 0; i < 20; ++i) {
            if (std::abs(f.positions[i]) <= options.home_tolerance) continue;
            if (!outside_tolerance) error += "；最后有效样本未到位关节：";
            else error += ", ";
            std::snprintf(detail, sizeof(detail), "%s_S%zu=%.9g rad", fingers[i / 4], i % 4 + 1, f.positions[i]);
            error += detail;
            outside_tolerance = true;
        }
        if (!outside_tolerance) error += "；最后有效样本所有关节均在容差内";
        if (!sent_at) error += "；尚未发布最终零目标";
        else if (f.position_ns <= sent_at) error += "；缺少零目标命令后的新鲜位置反馈";
        throw std::runtime_error(error);
    }
    // 即使停止确认失败也释放资源；错误不能被“已断开”状态吞掉。
    void shutdown(bool try_home, bool emergency) {
        std::string errors;
        if (try_home && phase == 2) {
            try { home(); } catch (const std::exception &e) { append(errors, e); emergency = true; }
        }
        if (connected && attempted) {
            stopped_confirmed = false;
            const auto before = monotonic_ns();
            try {
                transport->stop(emergency);
                wait_enabled(false, before);
                stopped_confirmed = true;
            } catch (const std::exception &e) {
                append(errors, e);
                if (!emergency) {
                    try {
                        const auto retry_at = monotonic_ns();
                        transport->stop(true);
                        wait_enabled(false, retry_at);
                        stopped_confirmed = true;
                    } catch (const std::exception &fallback) { append(errors, fallback); }
                }
            }
        }
        try { transport->release(); } catch (const std::exception &e) { append(errors, e); }
        connected = attempted = false;
        phase = fault.empty() && errors.empty() ? 4 : 3;
        if (!errors.empty()) {
            if (!fault.empty()) fault += "; ";
            fault += errors;
            fault_pending = true;
            throw LatchedFault(fault);
        }
    }
    void fail(const std::exception &e) {
        fault = e.what();
        fault_pending = true;
        try { shutdown(false, true); } catch (const std::exception &) {}
        phase = 3;
        throw LatchedFault(fault);
    }
    void enable() {
        require_ready();
        if (phase == 2) return;
        require(!approved.empty() && !preflight.empty() && identity.firmware == preflight && preflight == approved,
                "使能要求同设备官方只读预检的固件版本与显式审批版本完全相等");
        authorized = true;
        interrupt.store(false);
        try {
            // 配置或发布部分失败，也必须走实际停止确认路径。
            stopped_confirmed = false;
            attempted = true;
            // 用户显式enable接管指定手；先确认失能，再仅清错一次。
            const auto stopped_at = monotonic_ns();
            transport->stop(false);
            wait_enabled(false, stopped_at);
            const auto cleared_at = monotonic_ns();
            transport->clear_fault();
            const auto f = wait_cleared(cleared_at);
            validate(f.positions);
            transport->prepare(options);
            position = f.positions;
            velocity.fill(0);
            step_at = monotonic_ns();
            transport->send(position);
            const auto before = monotonic_ns();
            transport->enable();
            wait_enabled(true, before);
            phase = 2;
            if (options.auto_home) home();
            command_at = monotonic_ns();
            source_deadline_ns = 0;
        } catch (const std::exception &e) { fail(e); }
    }
    void check_source_deadline() {
        if (source_deadline_ns && monotonic_ns() >= source_deadline_ns) {
            const std::runtime_error expired("原始命令期限已到，直接停止并锁存");
            fail(expired);
        }
    }
    void send(const Angles &target, uint64_t deadline_ns) {
        require_ready();
        require(phase == 2, "必须先显式 enable");
        check_source_deadline();
        validate(target);
        try { sample(true); }
        catch (const std::exception &e) { fail(e); }
        if ((monotonic_ns() - command_at) * 1e-9 > options.command_timeout) {
            const std::runtime_error timeout("目标命令已超时，直接停止并拒绝迟到命令");
            fail(timeout);
        }
        try {
            track(target, deadline_ns);
            command_at = monotonic_ns();
            source_deadline_ns = deadline_ns;
        } catch (const std::exception &e) { fail(e); }
    }
    void disable(bool emergency) {
        if (!connected) {
            if (stopped_confirmed) return;
            if (!fault.empty()) throw LatchedFault(fault);
            return;
        }
        // 未显式通过固件审批并enable的只读会话，不得停止其他控制器。
        if (!authorized) return;
        try {
            stopped_confirmed = false;
            attempted = true;
            const auto before = monotonic_ns();
            transport->stop(emergency);
            wait_enabled(false, before);
            stopped_confirmed = true;
            attempted = false;
            source_deadline_ns = 0;
            phase = fault.empty() ? 1 : 3;
        } catch (const std::exception &e) { fail(e); }
    }
    void close() {
        if (phase == 2) check_source_deadline();
        if (connected) shutdown(options.auto_home && phase == 2, false);
        else if (fault.empty()) phase = 4;
        if (fault_pending) throw LatchedFault(fault);
    }
    void read(RealHandState &out) {
        require_ready();
        Feedback f;
        try { f = sample(phase == 2, phase != 2); }
        catch (const std::exception &e) { if (phase == 2) fail(e); throw; }
        out = {};
        std::copy(f.positions.begin(), f.positions.end(), out.positions);
        std::copy(f.velocities_rad_s.begin(), f.velocities_rad_s.end(), out.velocities_rad_s);
        std::copy(f.effort_a.begin(), f.effort_a.end(), out.effort_a);
        out.has_velocity = f.has_velocity;
        out.has_effort = f.has_effort;
        std::copy(identity.lower.begin(), identity.lower.end(), out.lower_limits);
        std::copy(identity.upper.begin(), identity.upper.end(), out.upper_limits);
        std::copy(f.enabled.begin(), f.enabled.end(), out.enabled);
        std::copy(f.online.begin(), f.online.end(), out.online);
        std::copy(f.errors.begin(), f.errors.end(), out.error_codes);
        out.timestamp_ns = f.position_ns;
        out.device_timestamp_us = f.device_us;
        out.phase = phase;
        message(out.firmware_version, sizeof(out.firmware_version), identity.firmware.c_str());
    }
    void watch() noexcept {
        while (!terminate.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::unique_lock<std::mutex> lock(control, std::try_to_lock);
            if (!lock || phase != 2) continue;
            try {
                check_source_deadline();
                sample(true);
            } catch (const LatchedFault &) {
                // 源期限检查已经停止并释放，不能再次走fail而重复释放。
                continue;
            } catch (const std::exception &e) {
                try { fail(e); } catch (...) {}
                continue;
            } catch (...) {
                try { std::runtime_error e("看门狗未知异常"); fail(e); } catch (...) {}
                continue;
            }
            if ((monotonic_ns() - command_at) * 1e-9 <= options.command_timeout) continue;
            try {
                throw std::runtime_error("目标命令超时，直接停止并锁存");
            } catch (const std::exception &e) {
                try { fail(e); } catch (...) {}
            } catch (...) {
                try { std::runtime_error e("看门狗未知异常"); fail(e); } catch (...) {}
            }
        }
    }
};

namespace {
template<class F> int call(RealHand *h, char *error, size_t size, F &&fn) noexcept {
    message(error, size, "");
    if (!h) { message(error, size, "空 RealHand 句柄"); return -1; }
    try {
        std::lock_guard<std::mutex> lock(h->control);
        try { fn(); }
        catch (const LatchedFault &) {
            // 只在C ABI确实交付该故障时消费；看门狗及无关参数错误不能消费它。
            h->fault_pending = false;
            throw;
        }
        return 0;
    }
    catch (const std::invalid_argument &e) { message(error, size, e.what()); return -1; }
    catch (const std::exception &e) { message(error, size, e.what()); return -2; }
    catch (...) { message(error, size, "原生控制未知异常"); return -2; }
}
}
extern "C" {
int real_hand_abi_version(void) { return 6; }
RealHand *real_hand_create(const RealHandOptions *o, char *error, size_t size) {
    message(error, size, "");
    try {
        if (!o) throw std::invalid_argument("空 RealHandOptions");
        return new RealHand(*o);
    } catch (const std::exception &e) { message(error, size, e.what()); }
    catch (...) { message(error, size, "原生构造未知异常"); }
    return nullptr;
}
int real_hand_connect(RealHand *h, char *e, size_t n) { return call(h, e, n, [&] { h->connect(); }); }
int real_hand_enable(RealHand *h, char *e, size_t n) { return call(h, e, n, [&] { h->enable(); }); }
int real_hand_send(RealHand *h, const double *p, size_t count,
                          uint64_t deadline_ns, char *e, size_t n) {
    return call(h, e, n, [&] {
        if (!p || count != 20) throw std::invalid_argument("目标必须包含20个弧度角");
        h->require_ready();
        if (!deadline_ns || monotonic_ns() >= deadline_ns) {
            const std::runtime_error expired("原始命令已过期，禁止发送");
            h->fail(expired);
        }
        Angles target; std::copy_n(p, 20, target.begin()); h->send(target, deadline_ns);
    });
}
int real_hand_read_state(RealHand *h, RealHandState *s, char *e, size_t n) {
    return call(h, e, n, [&] { if (!s) throw std::invalid_argument("空状态指针"); h->read(*s); });
}
int real_hand_home(RealHand *h, char *e, size_t n) {
    return call(h, e, n, [&] {
        h->require_ready(); require(h->phase == 2, "回零要求先使能");
        h->check_source_deadline();
        h->source_deadline_ns = 0;
        try { h->home(); h->command_at = monotonic_ns(); }
        catch (const std::exception &failure) { h->fail(failure); }
    });
}
int real_hand_disable(RealHand *h, char *e, size_t n) {
    if (h) h->interrupt.store(true);
    return call(h, e, n, [&] { h->disable(false); });
}
int real_hand_emergency_stop(RealHand *h, char *e, size_t n) {
    if (h) h->interrupt.store(true);
    return call(h, e, n, [&] { h->disable(true); });
}
int real_hand_close(RealHand *h, char *e, size_t n) { return call(h, e, n, [&] { h->close(); }); }
void real_hand_destroy(RealHand *h) { try { delete h; } catch (...) {} }
}
