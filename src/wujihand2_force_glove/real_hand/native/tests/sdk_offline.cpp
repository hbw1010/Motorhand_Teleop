#include "fake_sdk.h"
#include "real_hand.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
void expect(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint64_t command_deadline() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() + 5000000000ULL;
}
RealHandOptions options(int side = 1) {
    return {2, side, "离线设备身份", "", "offline-fw", "offline-fw", 1.5, 8, 3, .5, .2,
            200, .005, 1.5, .1, .04, 1.5, 3, .05, 20, 1};
}
struct Session {
    std::shared_ptr<fake_sdk::Scenario> scenario = std::make_shared<fake_sdk::Scenario>();
    RealHand *handle = nullptr;
    char error[1024]{};
    explicit Session(int side = 1, bool device_defaults = false) {
        scenario->hand = side;
        fake_sdk::enqueue(scenario);
        auto o = options(side);
        if (device_defaults) {
            o.max_velocity = o.max_acceleration = o.realtime_max_velocity = 0;
            o.effort_limit = o.mit_kp = o.mit_kd = -1;
            o.auto_home = 0;
        }
        handle = real_hand_create(&o, error, sizeof(error));
        expect(handle != nullptr, error);
    }
    ~Session() { real_hand_destroy(handle); }
    void connect() { expect(real_hand_connect(handle, error, sizeof(error)) == 0, error); }
    void enable() { expect(real_hand_enable(handle, error, sizeof(error)) == 0, error); }
    RealHandState read() {
        RealHandState state{};
        expect(real_hand_read_state(handle, &state, error, sizeof(error)) == 0, error);
        return state;
    }
    void close() { expect(real_hand_close(handle, error, sizeof(error)) == 0, error); }
};
void version_gate() {
    expect(real_hand_abi_version() == 6, "生产库必须完整切换到ABI6");
    fake_sdk::set_version("2026.10.12");
    const int initialized = fake_sdk::initialization_count();
    auto o = options();
    char error[512]{};
    RealHand *handle = real_hand_create(&o, error, sizeof(error));
    expect(handle != nullptr, error);
    const int result = real_hand_connect(handle, error, sizeof(error));
    real_hand_destroy(handle);
    fake_sdk::set_version("2026.8.31");
    expect(result != 0 && fake_sdk::initialization_count() == initialized,
           "SDK版本不匹配必须在init及任何连接之前拒绝");
}
void identity_and_order() {
    for (int side : {0, 1}) {
        Session s(side);
        for (int i = 0; i < 20; ++i) {
            s.scenario->positions[i] = static_cast<float>(i) * .001f;
            s.scenario->velocities_rad_s[i] = -.25f - static_cast<float>(i) * .125f;
            s.scenario->effort_a[i] = .0625f + static_cast<float>(i) * .03125f;
        }
        s.connect();
        const auto state = s.read();
        for (int i = 0; i < 20; ++i) {
            expect(state.positions[i] == static_cast<double>(static_cast<float>(i) * .001f),
                   "生产callback必须按NID重排全部关节而非使用反馈数组顺序");
            expect(state.has_velocity == 1 && state.has_effort == 1 &&
                   state.velocities_rad_s[i] == -.25 - i * .125 &&
                   state.effort_a[i] == .0625 + i * .03125,
                   "生产joint_states必须将真实非零速度/电流按同一NID映射传到ABI，不能使用诊断或位置替代");
            expect(state.online[i] == 1 && state.enabled[i] == 0, "生产callback必须传递实际在线与禁用状态");
        }
        expect(state.timestamp_ns != 0 && state.device_timestamp_us != 0, "接收与设备时间必须来自真实callback");
        expect(s.scenario->sends == 0 && !s.scenario->enabled, "SDKAdapter connect不得发送或使能");
        s.close();
        expect(s.scenario->closed_subs == 2 && s.scenario->releases == 1, "关闭必须join两条订阅后释放本手设备");
    }
    for (int fault = 0; fault < 4; ++fault) {
        Session s;
        if (fault == 0) s.scenario->serial = "错误序列号";
        if (fault == 1) s.scenario->hand = 0;
        if (fault == 2) s.scenario->supported = false;
        if (fault == 3) s.scenario->online_count = 19;
        expect(real_hand_connect(s.handle, s.error, sizeof(s.error)) != 0, "生产适配器必须拒绝身份与在线数异常");
        expect(s.scenario->sends == 0 && s.scenario->releases == 1, "身份失败不能写入设备且必须释放");
    }
}
void optional_feedback_unavailable_and_recovery() {
    // 固定typed SDK字段始终存在；NaN/Inf表示该量无法作为完整实测帧使用。
    for (bool missing_velocity : {false, true}) {
        Session s(1, true);
        for (int i = 0; i < 20; ++i) {
            s.scenario->velocities_rad_s[i] = -.5f - i * .125f;
            s.scenario->effort_a[i] = .125f + i * .03125f;
        }
        s.connect();
        const auto valid = s.read();
        expect(valid.has_velocity && valid.has_effort, "非零完整初始帧必须可用");
        const float unavailable = missing_velocity ? std::numeric_limits<float>::quiet_NaN()
                                                   : std::numeric_limits<float>::infinity();
        {
            std::lock_guard<std::mutex> lock(s.scenario->mutex);
            auto &values = missing_velocity ? s.scenario->velocities_rad_s : s.scenario->effort_a;
            values[17] = unavailable;
            s.scenario->positions[3] = .25f;
        }
        RealHandState invalid{};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        do {
            invalid = s.read();
            if (invalid.timestamp_ns > valid.timestamp_ns &&
                !(missing_velocity ? invalid.has_velocity : invalid.has_effort)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
        expect(invalid.has_velocity == !missing_velocity && invalid.has_effort == missing_velocity &&
               invalid.positions[3] == .25 && invalid.timestamp_ns > valid.timestamp_ns &&
               invalid.device_timestamp_us > valid.device_timestamp_us,
               "新帧可选量失效必须撤销旧可用标志并保留该帧真实位置及时间，不能锁存旧值");
        expect((missing_velocity ? invalid.effort_a[17] : invalid.velocities_rad_s[17]) ==
                   (missing_velocity ? .125 + 17 * .03125 : -.5 - 17 * .125),
               "一项可选量失效不能丢弃另一项完整反馈");
        {
            std::lock_guard<std::mutex> lock(s.scenario->mutex);
            s.scenario->velocities_rad_s[17] = -3.f;
            s.scenario->effort_a[17] = -.75f;
        }
        RealHandState recovered{};
        const auto recovery_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        do {
            recovered = s.read();
            if (recovered.has_velocity && recovered.has_effort &&
                recovered.velocities_rad_s[17] == -3. && recovered.effort_a[17] == -.75) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < recovery_deadline);
        expect(recovered.has_velocity && recovered.has_effort &&
               recovered.velocities_rad_s[17] == -3. && recovered.effort_a[17] == -.75,
               "后续完整帧必须恢复真实量，不能永久锁存可选量缺失");
        expect(s.scenario->sends == 0 && s.scenario->stops == 0 && !s.scenario->enabled,
               "可选量不可用仍是只读反馈，不得隐式使能或停止他人");
        s.close();
    }
}
void malformed_callbacks() {
    for (int fault = 1; fault <= 4; ++fault) {
        Session s;
        s.scenario->state_fault = fault;
        expect(real_hand_connect(s.handle, s.error, sizeof(s.error)) != 0,
               "位置callback必须拒绝缺失、重复、未知NID及非有限位置");
        expect(s.scenario->closed_subs == 2 && s.scenario->releases == 1, "错误callback后必须安全join释放");
    }
    for (int fault : {1, 2, 3, 4, 5, 7}) {
        Session s;
        s.scenario->diagnostic_fault = fault;
        expect(real_hand_connect(s.handle, s.error, sizeof(s.error)) != 0,
               "诊断callback必须拒绝缺失/重复NID、未知或矛盾使能状态及流错误");
        expect(s.scenario->sends == 0 && s.scenario->releases == 1, "反馈错误不得写入设备");
    }
    Session error_code;
    error_code.scenario->diagnostic_fault = 6;
    error_code.scenario->sticky_fault = true;
    error_code.connect();
    const auto state = error_code.read();
    expect(state.error_codes[0] == 0x3001 && state.error_codes[19] == 0, "错误码必须按关节语义真实传递");
    expect(real_hand_enable(error_code.handle, error_code.error, sizeof(error_code.error)) != 0 &&
           error_code.scenario->sends == 0, "无法清除的诊断故障必须阻止使能与目标发送");
    error_code.close();
}
void retarget_float_boundaries() {
    Session s;
    s.connect(); s.enable();
    const auto state = s.read();
    constexpr float thumb[4][2] = {{-1.187f,1.291f},{-1.484f,.698f},{-1.047f,1.570f},{-1.047f,1.570f}};
    constexpr float finger[4][2] = {{-1.047f,1.570f},{-.698f,.698f},{-1.047f,2.094f},{-1.047f,1.570f}};
    for (int i = 0; i < 20; ++i) {
        const auto &limit = i < 4 ? thumb[i] : finger[i % 4];
        expect(state.lower_limits[i] == static_cast<double>(limit[0]) &&
               state.upper_limits[i] == static_cast<double>(limit[1]), "驱动限位必须与retarget的float32值完全一致");
        for (int upper : {0, 1}) {
            double target[20]{};
            target[i] = static_cast<double>(limit[upper]);
            expect(real_hand_send(s.handle, target, 20, command_deadline(), s.error, sizeof(s.error)) == 0,
                   "retarget float32边界输出必须被同一send接口接受");
            const int sent = s.scenario->sends.load();
            target[i] = std::nextafter(target[i], upper ? std::numeric_limits<double>::infinity()
                                                       : -std::numeric_limits<double>::infinity());
            expect(real_hand_send(s.handle, target, 20, command_deadline(), s.error, sizeof(s.error)) == -1 && s.scenario->sends == sent,
                   "相邻越界double必须拒绝，不得用epsilon或裁剪掩盖");
        }
    }
    s.close();
}
void callback_timestamp_and_isolation() {
    Session paused;
    paused.connect();
    paused.scenario->pause = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (paused.scenario->paused_workers != 2 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    expect(paused.scenario->paused_workers == 2, "离线SDK反馈线程应确认暂停");
    const auto first = paused.read();
    const auto second = paused.read();
    expect(first.timestamp_ns == second.timestamp_ns, "生产Adapter读状态不能刷新接收时间");
    expect(first.device_timestamp_us == second.device_timestamp_us &&
           first.has_velocity == second.has_velocity && first.has_effort == second.has_effort &&
           std::equal(std::begin(first.velocities_rad_s), std::end(first.velocities_rad_s),
                      std::begin(second.velocities_rad_s)) &&
           std::equal(std::begin(first.effort_a), std::end(first.effort_a), std::begin(second.effort_a)),
           "缓存读不能更新同帧可选反馈或设备时间");
    paused.close();

    Session left(0), right(1);
    left.connect(); right.connect(); left.enable(); right.enable();
    const int right_stops = right.scenario->stops.load();
    left.scenario->send_failure = true;
    double target[20]{};
    target[0] = .1;
    expect(real_hand_send(left.handle, target, 20, command_deadline(), left.error, sizeof(left.error)) != 0,
           "真实生产发送路径必须传播SDK发送错误");
    expect(left.scenario->emergencies == 1 && left.scenario->closed_subs == 2 && left.scenario->releases == 1,
           "发送错误必须直接急停并释放本手SDK资源");
    expect(real_hand_send(right.handle, target, 20, command_deadline(), right.error, sizeof(right.error)) == 0,
           "单手故障不得影响另一手生产SDKAdapter");
    expect(right.scenario->stops == right_stops && right.scenario->releases == 0, "不得使用SDK全局释放影响另一手");
    right.close();
}
void emergency_requires_explicit_disable() {
    Session s;
    s.connect(); s.enable();
    const int sent = s.scenario->sends.load();
    expect(real_hand_emergency_stop(s.handle, s.error, sizeof(s.error)) == 0, s.error);
    const auto state = s.read();
    expect(std::all_of(std::begin(state.enabled), std::end(state.enabled),
                       [](uint8_t enabled) { return enabled == 0; }),
           "急停请求不保证失能，必须独立disable并以新鲜诊断确认");
    expect(!s.scenario->enabled && s.scenario->sends == sent, "急停不能发送回零目标");
    s.close();
}
void startup_recovery_and_fault_stop() {
    Session stale;
    stale.scenario->enabled = true;
    stale.scenario->diagnostic_fault = 6;
    stale.connect(); stale.enable();
    const auto recovered = stale.read();
    expect(std::all_of(std::begin(recovered.error_codes), std::end(recovered.error_codes),
                       [](uint32_t error) { return error == 0; }) &&
           std::all_of(std::begin(recovered.enabled), std::end(recovered.enabled),
                       [](uint8_t enabled) { return enabled == 1; }),
           "启动应接管残留使能、失能后清错并确认恢复，不能拒绝旧状态");
    expect(stale.scenario->clears == 1, "每次启动只能请求一次清错，不能循环掩盖硬件故障");
    stale.close();

    Session stopping;
    stopping.scenario->enable_fault = true;
    stopping.connect();
    expect(real_hand_enable(stopping.handle, stopping.error, sizeof(stopping.error)) != 0,
           "使能后ImmediateStop错误必须中止启动");
    expect(!stopping.scenario->enabled && stopping.scenario->emergencies > 0 &&
           stopping.scenario->sends == 1,
           "停止级故障应急停并失能，除使能前当前位置初始化外不得发送回零目标");
    stopping.close();
}
void wait_for_error_code(Session &s, uint16_t code) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    do {
        const auto state = s.read();
        if (state.error_codes[8] == code) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    throw std::runtime_error("消费者未收到语义关节8的新原始错误码");
}
void warning_startup_and_runtime() {
    Session s;
    s.scenario->diagnostic_code = fake_sdk::warning;
    s.scenario->sticky_fault = true;
    s.connect();
    expect(s.read().error_codes[8] == fake_sdk::warning, "连接态必须保留官方Warning原始码");
    s.enable();
    expect(s.read().error_codes[8] == fake_sdk::warning && s.scenario->clears == 1,
           "清错后仍存在的官方Warning不能阻止启动，也不能触发循环清错");
    // 换成另一个目录Warning、清零、再出现原Warning，均需按最新目录分类而非特殊放行码3。
    for (uint16_t code : {fake_sdk::synthetic_warning, uint16_t{0}, fake_sdk::warning}) {
        s.scenario->diagnostic_code = code;
        wait_for_error_code(s, code);
        const int sent = s.scenario->sends.load();
        double target[20]{};
        target[8] = .01;
        expect(real_hand_send(s.handle, target, 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
        expect(s.scenario->sends == sent + 1 && s.scenario->enabled && s.scenario->emergencies == 0,
               "Warning期间必须实际发布目标，不能吞掉发送或急停");
    }
    s.close();
}
void expect_safety_stop(Session &s, int sent) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    RealHandState state{};
    int result = 0;
    do {
        result = real_hand_read_state(s.handle, &state, s.error, sizeof(s.error));
        if (result != 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    expect(result != 0 && s.scenario->emergencies > 0 && !s.scenario->enabled &&
           s.scenario->sends == sent, "危险诊断必须立即急停失能，不能先回零或继续发送");
    double target[20]{};
    expect(real_hand_send(s.handle, target, 20, command_deadline(), s.error, sizeof(s.error)) != 0 &&
           s.scenario->sends == sent, "故障停机后必须拒绝新目标");
}
void warning_escalation_and_unknown_stop() {
    for (uint16_t code : {fake_sdk::deferred_stop, fake_sdk::immediate_stop, fake_sdk::synthetic_fatal,
                          fake_sdk::unknown_error, fake_sdk::synthetic_lookup_failure,
                          fake_sdk::synthetic_unknown_severity}) {
        Session s;
        s.scenario->diagnostic_code = fake_sdk::warning;
        s.scenario->sticky_fault = true;
        s.connect(); s.enable();
        const int sent = s.scenario->sends.load();
        s.scenario->diagnostic_code = code;
        expect_safety_stop(s, sent);
        s.close();
    }
}
void warning_does_not_mask_enable_loss() {
    Session s;
    s.scenario->diagnostic_code = fake_sdk::warning;
    s.scenario->sticky_fault = true;
    s.connect(); s.enable();
    const int sent = s.scenario->sends.load();
    s.scenario->enabled = false;
    expect_safety_stop(s, sent);
    s.close();
}
void stop_request_failures_are_preserved() {
    for (bool fail_disable : {false, true}) {
        Session s;
        s.connect(); s.enable();
        const int sent = s.scenario->sends.load();
        s.scenario->emergency_failure = !fail_disable;
        s.scenario->disable_failure = fail_disable;
        expect(real_hand_emergency_stop(s.handle, s.error, sizeof(s.error)) != 0,
               "任一停止请求失败不能被后续动作吞掉");
        expect(s.scenario->enabled == fail_disable && s.scenario->sends == sent,
               "急停失败仍须尝试失能；失能失败不能伪装成已停止或发送回零");
        s.close();
    }
}
void device_configuration_is_not_replaced_by_project_defaults() {
    for (bool defaults : {false, true}) {
        Session s(1, defaults);
        s.connect(); s.enable();
        expect(s.scenario->effort_writes == (defaults ? 0 : 1) &&
               s.scenario->mit_writes == (defaults ? 0 : 1),
               "默认不能写入项目自选电流/MIT参数；显式覆盖仍须生效");
        if (defaults) {
            double target[20]{};
            target[0] = .7;
            expect(real_hand_send(s.handle, target, 20, command_deadline(), s.error, sizeof(s.error)) == 0, s.error);
            {
                std::lock_guard<std::mutex> lock(s.scenario->mutex);
                expect(s.scenario->positions[0] == static_cast<float>(.7),
                       "无主机限速时SDK发布器必须收到未滤波原始目标");
            }
            expect(real_hand_home(s.handle, s.error, sizeof(s.error)) == 0, s.error);
        }
        const int before_close = s.scenario->sends;
        s.close();
        if (defaults) expect(s.scenario->sends == before_close, "显式生命周期close不得重复home");
    }
}
}
int main() {
    try {
        emergency_requires_explicit_disable();
        version_gate(); identity_and_order(); malformed_callbacks(); retarget_float_boundaries();
        optional_feedback_unavailable_and_recovery();
        callback_timestamp_and_isolation();
        startup_recovery_and_fault_stop(); stop_request_failures_are_preserved();
        warning_startup_and_runtime(); warning_escalation_and_unknown_stop();
        warning_does_not_mask_enable_loss();
        device_configuration_is_not_replaced_by_project_defaults();
        std::puts("生产C++核心及SDKAdapter离线回归通过：真实callback、身份、float32边界与双手隔离");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "SDKAdapter离线回归失败：%s\n", e.what()); return 1;
    }
}
