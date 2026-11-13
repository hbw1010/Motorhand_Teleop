#include "transport.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using real_hand::Angles;
using real_hand::Feedback;
using real_hand::Identity;
namespace {
void expect(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint64_t command_deadline() { return real_hand::monotonic_ns() + 5000000000ULL; }
struct Device {
    std::mutex mutex;
    Identity identity;
    Feedback feedback;
    std::atomic<bool> block_feedback{false}, feedback_entered{false};
    bool freeze = false, stale = false, send_failure = false, stop_failure = false;
    bool unknown = false, incomplete = false, enable_failure = false, release_failure = false;
    bool clear_ineffective = false, recur_on_enable = false, stop_ineffective = false, shift_on_clear = false;
    bool stale_clear_diagnostics = false, stale_clear_positions = false, stop_confirmed = false;
    bool stale_zero_positions = false, zero_feedback_confirmed = false;
    bool freeze_after_zero_feedback = false;
    int stale_zero_samples = 0, ignored_zero_commands = 0;
    int send_delay_ms = 0;
    double zero_step = 0;
    uint64_t zero_feedback_lag_ns = 0;
    double zero_bias = 0;
    uint64_t zero_sent_at = 0;
    uint64_t stop_at = 0;
    int sends = 0, enables = 0, stops = 0, emergencies = 0, releases = 0, clears = 0;
    Angles before_enable{};
    std::vector<Angles> history;
    std::vector<uint64_t> times;
    explicit Device(int hand) {
        identity.model = 2; identity.hand = hand;
        identity.serial = "offline"; identity.firmware = "approved";
        identity.lower.fill(-1); identity.upper.fill(1);
        feedback.positions.fill(.08);
        feedback.online.fill(1);
        feedback.complete = true;
    }
};
std::mutex factories_mutex;
std::deque<std::shared_ptr<Device>> factories;
class FakeTransport final : public real_hand::Transport {
    std::shared_ptr<Device> d;
public:
    explicit FakeTransport(std::shared_ptr<Device> value) : d(std::move(value)) {}
    Identity connect(const RealHandOptions &) override { return d->identity; }
    Feedback feedback() override {
        if (d->block_feedback.load()) {
            d->feedback_entered.store(true);
            while (d->block_feedback.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::lock_guard<std::mutex> lock(d->mutex);
        if (!d->stale) {
            const auto now = real_hand::monotonic_ns();
            const bool hold_zero_position = d->zero_sent_at &&
                (d->stale_zero_positions || d->stale_zero_samples > 0 ||
                 (d->freeze_after_zero_feedback && d->zero_feedback_confirmed));
            if (d->zero_sent_at && d->stale_zero_samples > 0) --d->stale_zero_samples;
            if ((!d->clears || !d->stale_clear_positions) && !hold_zero_position)
                d->feedback.position_ns = now - (d->zero_sent_at ? d->zero_feedback_lag_ns : 0);
            if (!d->clears || !d->stale_clear_diagnostics) d->feedback.diagnostics_ns = now;
        }
        if (d->stop_at && d->feedback.diagnostics_ns > d->stop_at &&
            std::all_of(d->feedback.enabled.begin(), d->feedback.enabled.end(), [](uint8_t v) { return v == 0; }))
            d->stop_confirmed = true;
        if (d->zero_sent_at && d->feedback.position_ns > d->zero_sent_at)
            d->zero_feedback_confirmed = true;
        auto result = d->feedback;
        if (d->unknown) result.enabled[19] = 7;
        if (d->incomplete) result.complete = false;
        return result;
    }
    void prepare(const RealHandOptions &) override {}
    void send(const Angles &target) override {
        std::lock_guard<std::mutex> lock(d->mutex);
        if (d->send_failure) throw std::runtime_error("注入真实运输发送失败");
        if (d->send_delay_ms)
            std::this_thread::sleep_for(std::chrono::milliseconds(d->send_delay_ms));
        ++d->sends;
        d->history.push_back(target); d->times.push_back(real_hand::monotonic_ns());
        const bool zero = std::all_of(target.begin(), target.end(), [](double v) { return v == 0; });
        if (zero && d->feedback.enabled[0]) {
            if (!d->zero_sent_at) d->zero_sent_at = real_hand::monotonic_ns();
            if (d->ignored_zero_commands > 0) {
                --d->ignored_zero_commands;
                return;
            }
            if (!d->freeze) {
                for (auto &position : d->feedback.positions)
                    position = d->zero_step > 0
                        ? position + std::clamp(d->zero_bias - position, -d->zero_step, d->zero_step)
                        : d->zero_bias;
            }
        } else if (!d->freeze) {
            d->feedback.positions = target;
        }
    }
    void enable() override {
        std::lock_guard<std::mutex> lock(d->mutex);
        d->before_enable = d->history.back();
        ++d->enables; d->feedback.enabled.fill(1);
        if (d->recur_on_enable) d->feedback.errors[0] = 3;
        if (d->enable_failure) {
            d->stop_failure = true;
            throw std::runtime_error("注入使能故障 Enc1BitRate");
        }
    }
    void stop(bool emergency) override {
        std::lock_guard<std::mutex> lock(d->mutex);
        ++d->stops; d->emergencies += emergency;
        if (d->stop_failure) throw std::runtime_error("注入停止失败");
        d->stop_at = real_hand::monotonic_ns();
        d->stop_confirmed = false;
        if (!d->stop_ineffective) d->feedback.enabled.fill(0);
    }
    void clear_fault() override {
        std::lock_guard<std::mutex> lock(d->mutex);
        expect(d->stop_confirmed, "必须先从新鲜反馈确认失能才能清错");
        ++d->clears;
        if (!d->clear_ineffective) d->feedback.errors.fill(0);
        if (d->shift_on_clear) d->feedback.positions.fill(.12);
    }
    void release() override {
        std::lock_guard<std::mutex> lock(d->mutex);
        ++d->releases;
        if (d->release_failure) throw std::runtime_error("注入释放失败");
    }
};
RealHandOptions options(int hand = 1) {
    return {2, hand, "offline", "", "approved", "approved", 1.5, 8, 3, .12, .08,
            200, .005, 1.5, .3, .04, 1.5, 3, .05, 20, 1};
}
struct Session {
    std::shared_ptr<Device> device;
    RealHand *handle = nullptr;
    char error[1024]{};
    explicit Session(int hand = 1, const char *preflight = "approved", double rate = 200,
                     double home_timeout = 1.5, double home_tolerance = .005,
                     bool auto_home = true, bool host_limits = true, double transition_duration = .04)
        : device(std::make_shared<Device>(hand)) {
        { std::lock_guard<std::mutex> lock(factories_mutex); factories.push_back(device); }
        auto o = options(hand);
        o.preflight_firmware_version = preflight;
        o.home_rate_hz = rate;
        o.home_timeout = home_timeout;
        o.home_tolerance = home_tolerance;
        o.transition_duration = transition_duration;
        o.auto_home = auto_home;
        if (!host_limits) o.max_velocity = o.max_acceleration = o.realtime_max_velocity = 0;
        handle = real_hand_create(&o, error, sizeof(error));
        expect(handle != nullptr, error);
    }
    ~Session() { real_hand_destroy(handle); }
    void connect() { expect(real_hand_connect(handle, error, sizeof(error)) == 0, error); }
    void enable() { expect(real_hand_enable(handle, error, sizeof(error)) == 0, error); }
    void close() { expect(real_hand_close(handle, error, sizeof(error)) == 0, error); }
    void move() {
        Angles target; target.fill(.2);
        for (int i = 0; i < 25; ++i) {
            expect(real_hand_send(handle, target.data(), 20, command_deadline(), error, sizeof(error)) == 0, error);
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
    }
    void wait_release() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < deadline) {
            { std::lock_guard<std::mutex> lock(device->mutex); if (device->releases) return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        throw std::runtime_error("独立C++看门狗没有释放资源");
    }
};
std::array<int, 6> actions(Session &s) {
    std::lock_guard<std::mutex> lock(s.device->mutex);
    return {s.device->sends, s.device->enables, s.device->stops, s.device->emergencies, s.device->releases, s.device->clears};
}
void expect_closed_without_actions(Session &s) {
    const auto before = actions(s);
    s.close();
    expect(s.error[0] == '\0', "成功close必须清除旧错误输出");
    s.close();
    expect(actions(s) == before, "故障已报告并释放后close不能新增发送、使能、停止或释放");
}
void expect_stops_without_actions(Session &s, bool confirmed) {
    const auto before = actions(s);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const int disabled = real_hand_disable(s.handle, s.error, sizeof(s.error));
        expect(disabled == (confirmed ? 0 : -2),
               "重复disable只能在已从新鲜反馈确认失能时成功");
        if (confirmed) expect(s.error[0] == '\0', "成功disable不能重报原控制故障");
        const int stopped = real_hand_emergency_stop(s.handle, s.error, sizeof(s.error));
        expect(stopped == (confirmed ? 0 : -2),
               "重复急停不能重报已安全收尾的控制故障，也不能吞掉未确认停止");
        if (confirmed) expect(s.error[0] == '\0', "成功急停不能重报原控制故障");
    }
    expect(actions(s) == before, "重复停止不能再次发送、使能、停止、清错或释放设备");
}
void fault_delivery() {
    for (bool during_enable : {true, false}) {
        Session s;
        s.connect();
        if (!during_enable) s.enable();
        {
            std::lock_guard<std::mutex> lock(s.device->mutex);
            s.device->enable_failure = during_enable;
            s.device->send_failure = !during_enable;
            s.device->stop_failure = !during_enable;
        }
        Angles target; target.fill(.2);
        const int result = during_enable
            ? real_hand_enable(s.handle, s.error, sizeof(s.error))
            : real_hand_send(s.handle, target.data(), target.size(), command_deadline(), s.error, sizeof(s.error));
        expect(result == -2, "同步运输故障必须越过C ABI边界返回原生失败");
        expect(std::strstr(s.error, during_enable ? "注入使能故障 Enc1BitRate" : "注入真实运输发送失败"),
               "首个同步异常必须保留原始故障");
        expect(std::strstr(s.error, "注入停止失败"), "首个同步异常必须保留实际停机失败");
        expect(s.device->emergencies == 1 && s.device->releases == 1, "同步故障必须尝试急停并释放");
        expect_stops_without_actions(s, false);
        expect_closed_without_actions(s);
        expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) == -2,
               "close幂等不能解除原故障锁存或允许重新使能");
        expect_closed_without_actions(s);
    }
}
void asynchronous_fault_delivery() {
    for (bool read_first : {false, true}) {
        Session s;
        s.connect(); s.enable();
        const int sends_before_fault = s.device->sends;
        {
            std::lock_guard<std::mutex> lock(s.device->mutex);
            s.device->feedback.errors[0] = 3;
            s.device->stop_failure = true;
        }
        s.wait_release();
        expect(s.device->sends == sends_before_fault, "看门狗反馈异常必须直接急停，不得回零发送");
        const auto before = actions(s);
        expect(real_hand_send(s.handle, nullptr, 0, command_deadline(), s.error, sizeof(s.error)) == -1,
               "无效调用仍必须返回参数错误");
        RealHandState state{};
        const int result = read_first
            ? real_hand_read_state(s.handle, &state, s.error, sizeof(s.error))
            : real_hand_close(s.handle, s.error, sizeof(s.error));
        expect(result == -2, "未报告的看门狗故障必须由首次有效调用报告，参数错误不能消费它");
        expect(s.device->emergencies == 1 && s.device->releases == 1,
               "看门狗反馈故障即使停止失败也必须尝试急停并释放");
        expect(actions(s) == before, "报告异步故障不能再次操作已释放设备");
        expect_closed_without_actions(s);
    }
}
void cleanup_fault_delivery() {
    Session s;
    s.connect();
    s.device->feedback.positions.fill(0);
    s.enable();
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->stop_failure = true;
        s.device->release_failure = true;
    }
    expect(real_hand_close(s.handle, s.error, sizeof(s.error)) == -2,
           "close中新发生的真实清理失败不能吞掉");
    expect(std::strstr(s.error, "注入停止失败") && std::strstr(s.error, "注入释放失败"),
           "close必须保留停止与资源释放的独立失败");
    expect(s.device->stops == 3 && s.device->emergencies == 1 && s.device->releases == 1,
           "正常停止失败仍必须尝试急停并释放");
    expect_closed_without_actions(s);
}
void confirmed_fault_cleanup() {
    for (bool watchdog : {false, true}) {
        Session s(1, "approved", 200, 1.5, .005, false, false);
        s.connect(); s.enable();
        Angles target; target.fill(.2);
        if (watchdog) {
            {
                std::lock_guard<std::mutex> lock(s.device->mutex);
                s.device->feedback.errors[0] = 3;
            }
            s.wait_release();
        } else {
            {
                std::lock_guard<std::mutex> lock(s.device->mutex);
                s.device->send_failure = true;
            }
            expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(),
                                  s.error, sizeof(s.error)) == -2,
                   "同步运动故障必须保留原失败返回");
            expect(std::strstr(s.error, "注入真实运输发送失败"), "不得把原控制故障替换为清理成功");
        }
        expect(s.device->stop_confirmed && s.device->emergencies == 1 && s.device->releases == 1,
               "自动故障收尾必须实际急停、确认新鲜失能并释放");
        expect_stops_without_actions(s, true);
        if (watchdog) {
            expect(real_hand_close(s.handle, s.error, sizeof(s.error)) == -2 &&
                   std::strstr(s.error, "关节报告停机故障"),
                   "成功停止不能消费尚未交付的看门狗故障，close仍须报告它");
        }
        const auto before = actions(s);
        RealHandState state{};
        expect(real_hand_read_state(s.handle, &state, s.error, sizeof(s.error)) == -2 &&
               real_hand_home(s.handle, s.error, sizeof(s.error)) == -2 &&
               real_hand_enable(s.handle, s.error, sizeof(s.error)) == -2 &&
               real_hand_send(s.handle, target.data(), target.size(), command_deadline(),
                              s.error, sizeof(s.error)) == -2,
               "安全收尾不能解除原控制故障锁存或重新控制设备");
        expect(actions(s) == before, "故障后控制调用只能拒绝，不能操作已释放设备");
        expect_stops_without_actions(s, true);
        expect_closed_without_actions(s);
    }
}
void stopped_evidence_lifecycle() {
    Session s(1, "approved", 200, 1.5, .005, false, false);
    s.connect(); s.enable();
    expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        // 连接仍存活时，设备可能被外部控制器重新使能；旧停止证据不能屏蔽急停。
        s.device->feedback.enabled.fill(1);
    }
    expect(real_hand_emergency_stop(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    expect(std::all_of(s.device->feedback.enabled.begin(), s.device->feedback.enabled.end(),
                       [](uint8_t value) { return value == 0; }),
           "连接存活时的显式急停必须重新停止并确认，不能复用旧失能证据");
    s.enable();
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->send_failure = true;
        s.device->stop_ineffective = true;
    }
    Angles target{};
    expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(),
                          s.error, sizeof(s.error)) == -2 &&
           std::strstr(s.error, "未从新鲜反馈确认停止"),
           "重新使能必须使旧失能证据失效，实际停止未生效必须报告");
    expect_stops_without_actions(s, false);
    expect_closed_without_actions(s);
}
void sides() {
    for (int hand : {0, 1}) {
        Session s(hand);
        s.connect();
        expect(s.device->sends == 0 && s.device->enables == 0 && s.device->stops == 0 && s.device->clears == 0,
               "connect必须只读");
        s.enable();
        expect(std::abs(s.device->before_enable[0] - .08) < 1e-12, "必须以实测位置初始化后才使能");
        expect(std::abs(s.device->feedback.positions[0]) <= .005, "启动必须实测回零");
        s.move();
        expect(s.device->feedback.positions[0] > .005, "运动目标应真实发往运输边界");
        s.close();
        expect(std::abs(s.device->feedback.positions[0]) <= .005, "正常close必须先回零");
        expect(s.device->stops == 2 && s.device->releases == 1 && s.device->emergencies == 0, "正常关闭顺序异常");
        s.close();
        expect(s.device->stops == 2 && s.device->releases == 1, "close必须幂等");
    }
}
void timeouts() {
    Session s;
    s.connect(); s.enable(); s.move();
    const int before = actions(s)[0];
    s.wait_release();
    std::lock_guard<std::mutex> lock(s.device->mutex);
    expect(s.device->sends == before && s.device->stops == 2, "命令超时不得发送回零目标");
    expect(s.device->emergencies == 1, "命令超时必须直接急停并锁存");
}
void measured_home() {
    Session s;
    s.device->freeze = true;
    s.connect();
    expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) != 0, "目标回零不能冒充实测回零");
    expect(s.device->emergencies == 1 && s.device->releases == 1, "回零失败必须急停释放");
    expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) != 0, "故障必须锁存");
}
void home_within_tolerance() {
    Session s(1, "approved", 200, 1.5, .1);
    s.device->identity.lower.fill(0);
    s.device->stale_zero_samples = 3;
    s.connect(); s.enable();
    const auto &history = s.device->history;
    const auto &times = s.device->times;
    expect(history.front()[0] == .08 &&
           std::all_of(history.back().begin(), history.back().end(), [](double v) { return v == 0; }),
           "初始位置已在容差内也必须发布明确20维零目标");
    expect(s.device->stale_zero_samples == 0 && s.device->zero_feedback_confirmed,
           "回零必须等到最终零目标后的新鲜位置，不能复用发送前反馈");
    Angles previous_velocity{};
    for (size_t i = 1; i < history.size(); ++i) {
        const double dt = std::min((times[i] - times[i - 1]) * 1e-9, .005);
        for (size_t joint = 0; joint < 20; ++joint) {
            const double velocity = (history[i][joint] - history[i - 1][joint]) / dt;
            expect(std::abs(velocity) <= 1.5 + .002, "大容差回零终点也必须限速");
            expect(std::abs(velocity - previous_velocity[joint]) <= 8 * dt + .002,
                   "到达精确零目标及停止时也必须受加速度约束");
            previous_velocity[joint] = velocity;
        }
    }
    expect(s.device->emergencies == 0 && s.device->releases == 0, "成功回零不能误急停");
    s.close();
}
void zero_home_feedback_failures() {
    for (bool stale : {false, true}) {
        Session s(1, "approved", 200, .6, .1);
        s.device->stale_zero_positions = stale;
        s.device->zero_bias = stale ? 0 : .11;
        s.connect();
        const auto started = std::chrono::steady_clock::now();
        expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) == -2,
               "零目标后位置超容差或只有发送前旧反馈都不能完成回零");
        expect(s.device->zero_sent_at != 0, "必须实际发布零目标后检验位置确认失败");
        if (!stale)
            expect(std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(600),
                   "零目标后真实偏差超容差必须等待回零截止时间，不能提前成功或误报其他故障");
        expect(s.device->emergencies == 1 && s.device->releases == 1,
               "零目标确认失败必须急停并释放");
        expect_closed_without_actions(s);
    }
}
void identity_feedback() {
    for (int failure = 0; failure < 5; ++failure) {
        Session s;
        if (failure == 0) s.device->identity.serial = "wrong";
        if (failure == 1) s.device->identity.hand = 0;
        if (failure == 2) s.device->identity.model = 99;
        if (failure == 3) s.device->incomplete = true;
        if (failure == 4) s.device->unknown = true;
        expect(real_hand_connect(s.handle, s.error, sizeof(s.error)) != 0, "错误身份/不完整/未知反馈必须拒绝");
        expect(s.device->enables == 0 && s.device->sends == 0 && s.device->releases == 1, "连接失败不能写设备");
    }
    Session s;
    s.device->identity.firmware = "different";
    s.connect();
    expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) != 0 &&
           s.device->sends == 0 && s.device->stops == 0 && s.device->clears == 0, "未审批固件不可操作设备");
    Session missing(1, "");
    missing.connect();
    expect(real_hand_enable(missing.handle, missing.error, sizeof(missing.error)) != 0 &&
           missing.device->sends == 0 && missing.device->stops == 0 && missing.device->clears == 0,
           "只读连接允许缺少预检版本，但使能必须拒绝");
}
void send_failure_and_isolation() {
    Session a(0), b(1);
    a.device->feedback.positions.fill(0);
    b.device->feedback.positions.fill(0);
    a.connect(); b.connect(); a.enable(); b.enable();
    { std::lock_guard<std::mutex> lock(a.device->mutex); a.device->send_failure = true; }
    const int before = a.device->sends;
    Angles target; target.fill(.2);
    expect(real_hand_send(a.handle, target.data(), 20, command_deadline(), a.error, sizeof(a.error)) != 0, "发送故障必须返回错误");
    expect(a.device->sends == before && a.device->emergencies == 1 && a.device->releases == 1,
           "发送故障必须直接急停，不能继续回零发送");
    expect(real_hand_send(b.handle, target.data(), 20, command_deadline(), b.error, sizeof(b.error)) == 0, b.error);
    expect(b.device->stops == 1 && b.device->releases == 0, "单手故障不能停止或释放另手");
    b.close();
}
void rejection_and_stop() {
    Session s;
    s.connect(); s.enable();
    Angles target; target.fill(2);
    const int before = s.device->sends;
    expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == -1, "越界角度必须拒绝");
    expect(s.device->sends == before, "越界角不能裁剪发送");
    expect(real_hand_emergency_stop(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->sends == before && s.device->emergencies == 1, "显式急停不能回零");
    s.close();
    Session broken;
    broken.connect(); broken.enable();
    { std::lock_guard<std::mutex> lock(broken.device->mutex); broken.device->stop_failure = true; }
    expect(real_hand_disable(broken.handle, broken.error, sizeof(broken.error)) != 0, "停止失败不能报告成功");
    expect_closed_without_actions(broken);
    expect(broken.device->releases == 1, "停止失败也必须释放资源");
}
void receive_timestamp() {
    Session s;
    s.connect();
    RealHandState first{}, second{};
    { std::lock_guard<std::mutex> lock(s.device->mutex); s.device->stale = true; }
    expect(real_hand_read_state(s.handle, &first, s.error, sizeof(s.error)) == 0, s.error);
    expect(real_hand_read_state(s.handle, &second, s.error, sizeof(s.error)) == 0, s.error);
    expect(first.timestamp_ns == second.timestamp_ns, "read_state不能刷新主机接收时间");
    expect(!first.has_velocity && !first.has_effort && !second.has_velocity && !second.has_effort,
           "运输层未提供可选反馈时ABI必须明确不可用，不把默认数组当成实测零值");
    s.close();
    Session lost;
    lost.connect(); lost.enable();
    const int before = lost.device->sends;
    { std::lock_guard<std::mutex> lock(lost.device->mutex); lost.device->stale = true; }
    lost.wait_release();
    expect(lost.device->emergencies == 1 && lost.device->sends == before, "反馈超时必须直接急停，不得回零");
}
void motion_limits() {
    Session s;
    s.connect(); s.enable();
    const size_t realtime_begin = s.device->history.size();
    s.move();
    const size_t home_begin = s.device->history.size();
    s.close();
    const auto &history = s.device->history;
    const auto &times = s.device->times;
    for (size_t i = 1; i < history.size(); ++i) {
        const double elapsed = (times[i] - times[i - 1]) * 1e-9;
        const double limit = i >= realtime_begin && i < home_begin ? 3.0 : 1.5;
        for (size_t joint = 0; joint < 20; ++joint) {
            if (std::abs(history[i][joint] - history[i - 1][joint]) > limit * elapsed + 1e-5) {
                char detail[256];
                std::snprintf(detail, sizeof(detail),
                              "发布角度变化超过限速：sample=%zu joint=%zu delta=%.9g bound=%.9g dt=%.9g phase=%s",
                              i, joint, std::abs(history[i][joint] - history[i - 1][joint]),
                              limit * elapsed + 1e-5, elapsed,
                              i >= realtime_begin && i < home_begin ? "realtime" : "home");
                throw std::runtime_error(detail);
            }
            expect(history[i][joint] >= -1 && history[i][joint] <= 1, "发布角度越过硬限位");
        }
    }
    // 启动从静止开始，加速度约束必须比仅限速更严格。
    const double dt = std::min((times[1] - times[0]) * 1e-9, .005);
    expect(std::abs(history[1][0] - history[0][0]) <= 8 * dt * dt + 1e-5,
           "启动第一步必须受加速度约束");
    double previous_velocity = 0;
    for (size_t i = home_begin; i < history.size(); ++i) {
        const double dt = std::min((times[i] - times[i - 1]) * 1e-9, .005);
        const double velocity = (history[i][0] - history[i - 1][0]) / dt;
        expect(std::abs(velocity - previous_velocity) <= 8 * dt + .002,
               "实时切回回零后必须重新遵守回零加速度约束");
        previous_velocity = velocity;
    }
}
void delayed_publish_does_not_credit_motion_time() {
    Session s(1, "approved", 200, 1.5, .005, false, true);
    s.connect(); s.enable();
    Angles target; target.fill(.7);
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->send_delay_ms = 20;
    }
    expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(),
                          s.error, sizeof(s.error)) == 0, s.error);
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->send_delay_ms = 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(),
                          s.error, sizeof(s.error)) == 0, s.error);
    const size_t last = s.device->history.size() - 1;
    const double elapsed = (s.device->times[last] - s.device->times[last - 1]) * 1e-9;
    expect(std::abs(s.device->history[last][0] - s.device->history[last - 1][0]) <= 3.0 * elapsed + 1e-5,
           "前一帧发送耗时不得变成下一帧的额外运动时间，导致实际发令超过显式限速");
    expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    s.close();
}
void realtime_progress_independent_of_home_rate() {
    // 回零频率不得改变50Hz外部命令的物理时间尺度。
    for (double home_rate : {50.0, 200.0}) {
        Session s(0, "approved", home_rate);
        s.connect(); s.enable();
        uint64_t previous_begin = s.device->times.back();
        Angles previous = s.device->history.back();
        Angles target; target.fill(.5);
        const auto start = std::chrono::steady_clock::now();
        for (int tick = 1; tick <= 9; ++tick) {
            std::this_thread::sleep_until(start + std::chrono::milliseconds(20 * tick));
            const auto begin = real_hand::monotonic_ns();
            expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
            const auto end = real_hand::monotonic_ns();
            // 积分时刻在ABI调用内部，运输回调时间不能冒充精确积分时刻。
            const double interval_bound = (end - previous_begin) * 1e-9;
            const auto &current = s.device->history.back();
            for (size_t joint = 0; joint < 20; ++joint)
                expect(std::abs(current[joint] - previous[joint]) <= 3 * interval_bound + 1e-12,
                       "实时响应不得绕过独立速度上限");
            previous_begin = begin;
            previous = current;
        }
        expect(s.device->history.back()[0] >= .45,
               "旧实时模式的0.5rad阶跃应在180ms内推进90%，不能套用回零加速度轨迹");
        expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
        s.close();
    }
}
void realtime_direct_and_reset() {
    Session s;
    s.connect(); s.enable();
    Angles target{};
    target[0] = .004;
    for (int i = 0; i < 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
    }
    expect(s.device->history.back()[0] == .004, "不得附加旧项目0.5度死区或自适应滤波");
    target[0] = 1;
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
    }
    const double before = s.device->history.back()[0];
    expect(before > .99 && before <= 1, "实时目标应能到达限位附近且不能越界");
    target[0] = -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->history.back()[0] < before, "反向实时目标不得继续按旧加速度轨迹朝限位推进");
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    target.fill(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->history.back()[0] == 0, "回零后零目标不得使关节再次运动");
    s.close();
}
void explicit_lifecycle_and_default_position_path() {
    Session s(1, "approved", 200, 1.5, .005, false, false);
    s.connect();
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == -2,
           "未使能不能回零");
    s.enable();
    expect(s.device->sends == 1 && s.device->feedback.positions[0] == .08,
           "auto_home=0的enable只发布当前位置保持，不能隐式回零");
    s.device->stale_zero_samples = 3;
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->zero_feedback_confirmed && s.device->stale_zero_samples == 0,
           "默认position回零必须等待零目标命令后的新鲜反馈");
    Angles target; target.fill(.7);
    expect(real_hand_send(s.handle, target.data(), 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->history.back() == target, "未显式限速时必须原样发布目标");
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    const int after_home = s.device->sends;
    expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    s.close();
    expect(s.device->sends == after_home && s.device->enables == 1,
           "显式home→disable→close不得重复回零或重新使能");

    Session direct_close(1, "approved", 200, 1.5, .005, false, false);
    direct_close.connect(); direct_close.enable();
    const int before_close = direct_close.device->sends;
    direct_close.close();
    expect(direct_close.device->sends == before_close, "auto_home=0的close即使仍使能也不得回零");
}
void continuous_default_home() {
    for (bool gradual : {false, true}) {
        Session s(1, "approved", 200, .2, .005, false, false, .04);
        s.connect(); s.enable();
        const int before = s.device->sends;
        {
            std::lock_guard<std::mutex> lock(s.device->mutex);
            s.device->ignored_zero_commands = 1;
            s.device->stale_zero_samples = 2;
            s.device->zero_step = gradual ? .02 : 0;
            s.device->zero_feedback_lag_ns = gradual ? 10000000 : 0;
        }
        expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0,
               "默认回零必须持续发零，容忍首帧遗漏并等待需要连续命令的设备实测收敛");
        expect(s.device->sends > before + 1 && s.device->zero_feedback_confirmed &&
               s.device->feedback.positions[0] <= .005,
               "回零成功必须来自重复命令后的真实到位和首次零命令后的新鲜反馈");
        bool intermediate = false, final_zero = false;
        for (size_t i = static_cast<size_t>(before); i < s.device->history.size(); ++i) {
            const double value = s.device->history[i][0];
            intermediate = intermediate || (value > 0 && value < .08);
            final_zero = final_zero || value == 0;
            expect(!final_zero || value == 0, "最终零目标发布后必须持续保持零位");
            if (i > static_cast<size_t>(before))
                expect(s.device->times[i] - s.device->times[i - 1] >= 5000000,
                       "渐变与重发零目标都必须遵守现有home_rate_hz周期");
        }
        expect(intermediate && final_zero, "默认回零必须经过中间目标再到最终零位");
        expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
        s.close();
    }
}
void quintic_home_from_measured_pose() {
    constexpr double duration = .12;
    // 初始实测值就在宽容差内，也必须完整走完轨迹而不是立即成功。
    Session s(1, "approved", 200, .8, .5, false, false, duration);
    s.connect(); s.enable();
    Angles start{};
    for (size_t i = 0; i < start.size(); ++i) start[i] = i % 2 ? -.4 : .4;
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->feedback.positions = start;
        s.device->stale_zero_samples = 2;
    }
    const size_t before = s.device->history.size();
    const auto began = real_hand::monotonic_ns();
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    const auto &history = s.device->history;
    const auto &times = s.device->times;
    expect(s.device->zero_sent_at - began >= static_cast<uint64_t>(duration * 1e9),
           "宽容差不能跳过完整transition_duration");
    expect(s.device->zero_feedback_confirmed && s.device->stale_zero_samples == 0,
           "轨迹完成仍须等待最终零命令后的新鲜实测反馈");
    expect(std::abs(history[before][0] - start[0]) < .001,
           "轨迹必须从本次有效实测姿态开始，而非旧命令或零位");
    size_t intermediate = 0, final_zero = 0;
    for (size_t i = before; i < history.size(); ++i) {
        if (history[i][0] > 0 && history[i][0] < start[0]) ++intermediate;
        if (history[i][0] == 0) ++final_zero;
        // 轨迹计时起点在home入口后、首帧发布前；运输时间界定可观测曲线范围。
        const auto weight = [](double u) {
            u = std::clamp(u, 0.0, 1.0);
            return u * u * u * (10 + u * (-15 + 6 * u));
        };
        const double earliest = (times[i] - times[before]) * 1e-9 / duration;
        const double latest = (times[i] - began) * 1e-9 / duration;
        const double lower = start[0] * (1 - weight(latest));
        const double upper = start[0] * (1 - weight(earliest));
        expect(history[i][0] >= lower - .001 && history[i][0] <= upper + .001,
               "中间目标必须遵守固定起点五次smoothstep，不能线性或递归滤波");
        for (size_t joint = 0; joint < start.size(); ++joint) {
            expect(std::abs(history[i][joint]) <= std::abs(start[joint]),
                   "正负关节轨迹必须保持在起点与零位之间");
            expect(std::abs(history[i][joint] / start[joint] - history[i][0] / start[0]) < 1e-12,
                   "所有关节必须共用同一轨迹进度");
            if (i > before)
                expect(std::abs(history[i][joint]) <= std::abs(history[i - 1][joint]),
                       "无额外主机限速的回零参考必须单调接近零");
        }
    }
    expect(intermediate > 2 && final_zero >= 2, "必须发布连续中间目标并保持最终零位等待新鲜反馈");
    Angles target; target.fill(.7);
    expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(),
                          s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->history.back() == target, "回零渐变不能泄漏到正常遥操send");
    expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    s.close();
}
void transition_timeout_within_tolerance() {
    Session s(1, "approved", 200, .03, .1, false, false, .12);
    s.connect(); s.enable();
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == -2,
           "home_timeout短于轨迹时即使实测一直在容差内也必须超时");
    expect(std::strstr(s.error, "尚未发布最终零目标") &&
           std::strstr(s.error, "最后有效样本所有关节均在容差内"),
           "轨迹未结束超时必须区分容差内实测与尚未发出的最终目标");
    expect(s.device->zero_sent_at == 0 && s.device->emergencies == 1 && s.device->releases == 1,
           "轨迹超时不能自动扩大期限或跳零，必须停机释放");
    expect_closed_without_actions(s);
}
void transition_duration_validation() {
    for (double duration : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        auto o = options();
        o.transition_duration = duration;
        char error[256]{};
        auto *handle = real_hand_create(&o, error, sizeof(error));
        if (handle) real_hand_destroy(handle);
        expect(!handle && std::strstr(error, "transition_duration"),
               "原生ABI必须在创建运输或连接前拒绝非正/非有限过渡时长");
    }
}
void constrained_home_keeps_first_final_zero_baseline() {
    Session s(1, "approved", 200, 1.5, .005, false, true, .005);
    s.connect(); s.enable();
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->freeze_after_zero_feedback = true;
    }
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0,
           "显式限速回零必须保留首次最终零命令的新鲜度基准，不能被速度归零帧推迟");
    expect(s.device->zero_feedback_confirmed &&
           s.device->feedback.position_ns < s.device->times.back(),
           "回归必须使用首次最终零之后、后续零速保持帧之前的有效实测反馈");
    expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    s.close();
}
void home_timeout_diagnostics() {
    Session s(1, "approved", 200, .03, .1, false, false, .005);
    s.connect(); s.enable();
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->freeze = true;
        s.device->feedback.positions.fill(.09);
        s.device->feedback.positions[0] = .11;
        s.device->feedback.positions[5] = -.12;
        s.device->feedback.positions[10] = .15;
        s.device->feedback.positions[15] = -.16;
        s.device->feedback.positions[19] = .2;
    }
    expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == -2,
           "真实未到位的关节必须在回零截止时间失败");
    for (const auto *detail : {"回零超时", "home_tolerance=0.1", "timeout=0.03",
                               "thumb_S1=0.11 rad", "index_S2=-0.12 rad",
                               "middle_S3=0.15 rad", "ring_S4=-0.16 rad", "pinky_S4=0.2 rad"})
        expect(std::strstr(s.error, detail), "回零超时必须列出容差、超时及每个未到位关节的官方名称和实测rad");
    expect(!std::strstr(s.error, "thumb_S2="), "超时诊断不能把容差内的关节列为未到位");
    expect_stops_without_actions(s, true);
    expect_closed_without_actions(s);
}
void default_home_failures_never_resume_or_rehome() {
    for (bool stale : {false, true}) {
        Session s(1, "approved", 200, .03, .005, false, false, .005);
        s.connect(); s.enable();
        {
            std::lock_guard<std::mutex> lock(s.device->mutex);
            s.device->freeze = !stale;
            s.device->stale_zero_positions = stale;
        }
        expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == -2,
               "默认位置回零也必须拒绝未到位或命令前旧位置反馈");
        expect(std::strstr(s.error, "home_tolerance=0.005") && std::strstr(s.error, "timeout=0.03"),
               "回零超时必须报告实际容差及超时参数");
        if (stale)
            expect(std::strstr(s.error, "缺少零目标命令后的新鲜位置反馈"),
                   "数值已到位但只有旧样本时必须单独说明缺少命令后新鲜反馈");
        expect(s.device->emergencies == 1 && s.device->releases == 1,
               "默认回零失败必须直接停机并释放，不得自动重使能");
        expect_stops_without_actions(s, true);
        expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == -2,
               "回零故障必须锁存，不允许再次home");
        expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) == -2,
               "回零故障必须锁存，不允许自动重新使能");
        expect_closed_without_actions(s);
    }
}
void bounded_home_wait(bool host_limits) {
    Session timeout(1, "approved", .01, .03, .005, true, host_limits, .005);
    timeout.device->freeze = true;
    timeout.connect();
    auto started = std::chrono::steady_clock::now();
    expect(real_hand_enable(timeout.handle, timeout.error, sizeof(timeout.error)) != 0,
           "低发送频率也必须遵守回零超时");
    expect(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(500),
           "回零等待不能睡过deadline");
    expect(timeout.device->sends <= 2 && timeout.device->emergencies == 1,
           "等待分片不能提高发送频率，超时必须急停");

    Session interrupted(1, "approved", .01, 2, .005, true, host_limits);
    interrupted.device->freeze = true;
    interrupted.connect();
    int enabled_result = 0;
    std::thread enabling([&] {
        enabled_result = real_hand_enable(interrupted.handle, interrupted.error, sizeof(interrupted.error));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    bool enabled = false;
    while (std::chrono::steady_clock::now() < deadline) {
        { std::lock_guard<std::mutex> lock(interrupted.device->mutex); enabled = interrupted.device->enables != 0; }
        if (enabled) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    started = std::chrono::steady_clock::now();
    char stop_error[512]{};
    const int stop_result = real_hand_emergency_stop(interrupted.handle, stop_error, sizeof(stop_error));
    enabling.join();
    expect(enabled && enabled_result != 0, "显式停止必须中断回零操作");
    expect(stop_result == 0 && stop_error[0] == '\0',
           "中断触发的故障已确认失能后，发起中断的急停调用应幂等成功");
    expect(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(500),
           "低频回零等待必须及时响应显式急停");
    expect(interrupted.device->sends <= 2 && interrupted.device->releases == 1,
           "中断后不得继续发送，且须释放本手资源");
}
void startup_recovery() {
    Session s;
    s.device->feedback.enabled.fill(1);
    s.device->feedback.errors[8] = 3;
    s.device->shift_on_clear = true;
    s.connect();
    expect(s.device->stops == 0 && s.device->clears == 0 && s.device->enables == 0 && s.device->sends == 0,
           "带残留使能及旧错误的connect仍必须严格只读");
    s.enable();
    expect(s.device->stops == 1 && s.device->clears == 1 && s.device->enables == 1,
           "显式enable必须先确认失能、只清错一次，再使能");
    expect(std::abs(s.device->before_enable[0] - .12) < 1e-12 &&
           std::abs(s.device->feedback.positions[0]) <= .005, "清错启动必须从实测当前位置初始化并回零");
    s.close();
}
void startup_recovery_failures() {
    for (int failure = 0; failure < 5; ++failure) {
        Session s;
        s.device->feedback.enabled.fill(1);
        s.device->feedback.errors[8] = 3;
        s.device->clear_ineffective = failure == 0;
        s.device->recur_on_enable = failure == 1;
        s.device->stale_clear_diagnostics = failure == 2;
        s.device->stale_clear_positions = failure == 3;
        s.device->stop_ineffective = failure == 4;
        s.connect();
        expect(real_hand_enable(s.handle, s.error, sizeof(s.error)) == -2,
               "清错无效、复发、旧反馈或无法确认失能都必须拒绝启动");
        expect(s.device->clears == (failure == 4 ? 0 : 1), "失能确认前不可清错，失败后不可重试清错");
        expect(s.device->enables == (failure == 1 ? 1 : 0) && s.device->sends == (failure == 1 ? 1 : 0),
               "清错未确认不可使能或发送；使能后复发只允许此前初始化，不得回零");
        expect(s.device->emergencies == 1 && s.device->releases == 1, "启动恢复失败必须急停并释放");
        expect_closed_without_actions(s);
    }
}
void warning_feedback() {
    Session s;
    s.device->feedback.errors[8] = 3;
    s.device->feedback.warnings[8] = true;
    s.device->clear_ineffective = true;
    s.connect(); s.enable();
    RealHandState state{};
    expect(real_hand_read_state(s.handle, &state, s.error, sizeof(s.error)) == 0 &&
           state.error_codes[8] == 3 && state.enabled[8] == 1,
           "官方Warning必须保留原始码并允许使能后读取");
    Angles target; target.fill(.2);
    expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(), s.error, sizeof(s.error)) == 0,
           "官方Warning不能阻断正常控制");
    expect(s.device->emergencies == 0 && s.device->releases == 0 && s.device->clears == 1,
           "持续Warning不能触发急停、释放或重复清错");
    const int before = s.device->sends;
    {
        std::lock_guard<std::mutex> lock(s.device->mutex);
        s.device->feedback.errors[9] = 0xffff;
    }
    expect(real_hand_send(s.handle, target.data(), target.size(), command_deadline(), s.error, sizeof(s.error)) == -2,
           "Warning不能掩盖其他关节的未知或停机故障");
    expect(s.device->sends == before && s.device->emergencies == 1 && s.device->releases == 1,
           "混合Warning与未知错误必须直接急停，不得继续发送");
    expect_closed_without_actions(s);
}
void active_feedback_fault() {
    for (bool read_state : {false, true}) {
        Session s;
        s.connect(); s.enable();
        const int before = s.device->sends;
        { std::lock_guard<std::mutex> lock(s.device->mutex); s.device->feedback.errors[8] = 3; }
        Angles target; target.fill(.2);
        RealHandState state{};
        const int result = read_state
            ? real_hand_read_state(s.handle, &state, s.error, sizeof(s.error))
            : real_hand_send(s.handle, target.data(), target.size(), command_deadline(), s.error, sizeof(s.error));
        expect(result == -2, "未分类为Warning的运行时故障必须立即返回错误");
        expect(s.device->sends == before && s.device->emergencies == 1 && s.device->releases == 1 &&
               s.device->clears == 1, "运行时反馈故障必须直接急停，不得回零或再次清错");
        expect_closed_without_actions(s);
    }
}
void readonly_close_does_not_take_ownership() {
    Session s;
    s.device->feedback.enabled.fill(1);
    s.connect();
    s.close();
    const auto counts = actions(s);
    expect(counts[0] == 0 && counts[1] == 0 && counts[2] == 0 && counts[3] == 0 &&
           counts[4] == 1 && counts[5] == 0,
           "未enable的只读连接关闭不得接管、失能、清错或回零外部已使能设备");
}
void readonly_stop_does_not_take_ownership() {
    Session s;
    s.device->feedback.enabled.fill(1);
    s.connect();
    expect(real_hand_disable(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    expect(real_hand_emergency_stop(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    expect(s.device->stops == 0 && s.device->emergencies == 0 &&
           s.device->feedback.enabled[0] == 1,
           "只读会话的停止请求不得取得他人设备的控制权");
    s.close();
}
void readonly_destroy_does_not_take_ownership() {
    std::shared_ptr<Device> device;
    {
        Session s;
        device = s.device;
        device->feedback.enabled.fill(1);
        s.connect();
    }
    expect(device->sends == 0 && device->enables == 0 && device->stops == 0 &&
           device->emergencies == 0 && device->clears == 0 && device->releases == 1,
           "只读实例析构只能释放连接，不得接管外部已使能设备");
}
void original_source_deadlines() {
    {
        Session s;
        s.connect(); s.enable();
        Angles target; target.fill(.2);
        const auto deadline = real_hand::monotonic_ns() + 40000000;
        expect(real_hand_send(s.handle, target.data(), 20, deadline, s.error, sizeof(s.error)) == 0,
               s.error);
        const int before = actions(s)[0];
        s.wait_release();
        expect(actions(s)[0] == before && actions(s)[3] == 1,
               "原始源期限到后必须直接停，不能等待command_timeout再回零");
        expect(real_hand_send(s.handle, target.data(), 20, real_hand::monotonic_ns() + 100000000,
                                     s.error, sizeof(s.error)) == -2,
               "新期限不能解除原始源过期锁存");
        expect_closed_without_actions(s);
    }
    {
        Session s;
        s.connect(); s.enable();
        s.device->block_feedback.store(true);
        std::thread reader([&] {
            RealHandState state{};
            char error[1024]{};
            real_hand_read_state(s.handle, &state, error, sizeof(error));
        });
        while (!s.device->feedback_entered.load()) std::this_thread::yield();
        Angles target; target.fill(.2);
        const int before = actions(s)[0];
        const auto deadline = real_hand::monotonic_ns() + 10000000;
        int result = 0;
        std::thread sender([&] {
            result = real_hand_send(s.handle, target.data(), 20, deadline, s.error, sizeof(s.error));
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        s.device->block_feedback.store(false);
        reader.join(); sender.join();
        expect(result == -2 && actions(s)[0] == before && actions(s)[3] == 1,
               "在原生SDK控制锁等待期间过期的目标绝不能发布");
        expect_closed_without_actions(s);
    }
}
}
namespace real_hand {
std::unique_ptr<Transport> make_transport() {
    std::lock_guard<std::mutex> lock(factories_mutex);
    if (factories.empty()) throw std::runtime_error("离线运输队列为空，禁止回退真实SDK");
    auto device = factories.front(); factories.pop_front();
    return std::make_unique<FakeTransport>(std::move(device));
}
}
int main() {
    try {
        startup_recovery(); startup_recovery_failures(); warning_feedback(); active_feedback_fault();
        fault_delivery(); asynchronous_fault_delivery(); cleanup_fault_delivery();
        confirmed_fault_cleanup(); stopped_evidence_lifecycle();
        sides(); timeouts(); measured_home(); home_within_tolerance(); zero_home_feedback_failures(); identity_feedback();
        send_failure_and_isolation(); rejection_and_stop(); receive_timestamp(); motion_limits();
        bounded_home_wait(false); bounded_home_wait(true);
        delayed_publish_does_not_credit_motion_time();
        realtime_progress_independent_of_home_rate(); realtime_direct_and_reset();
        explicit_lifecycle_and_default_position_path();
        continuous_default_home(); home_timeout_diagnostics();
        quintic_home_from_measured_pose(); transition_timeout_within_tolerance();
        transition_duration_validation();
        constrained_home_keeps_first_final_zero_baseline();
        default_home_failures_never_resume_or_rehome();
        original_source_deadlines();
        readonly_close_does_not_take_ownership();
        readonly_stop_does_not_take_ownership();
        readonly_destroy_does_not_take_ownership();
        std::puts("生产C++状态机离线回归通过：二代双侧、回零、故障、身份、反馈、双手隔离");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "离线回归失败：%s\n", e.what()); return 1;
    }
}
