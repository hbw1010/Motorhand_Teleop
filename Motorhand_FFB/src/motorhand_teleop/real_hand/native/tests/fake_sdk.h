#pragma once
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace fake_sdk {
// 前三项与固定官方目录一致；synthetic_*仅为覆盖目录分类边界的离线条目。
inline constexpr uint16_t warning = 0x0003;
inline constexpr uint16_t deferred_stop = 0x1001;
inline constexpr uint16_t immediate_stop = 0x2001;
inline constexpr uint16_t synthetic_fatal = 0xf001;
inline constexpr uint16_t synthetic_unknown_severity = 0xf002;
inline constexpr uint16_t synthetic_lookup_failure = 0xf003;
inline constexpr uint16_t synthetic_warning = 0xf004;
inline constexpr uint16_t unknown_error = 0xffff;
// 仅测试程序显式排队；共享库没有真实SDK依赖，也不存在设备回退。
struct Scenario {
    std::mutex mutex;
    std::string serial;
    int hand = 1;
    bool supported = true;
    uint8_t online_count = 20;
    std::array<float, 20> positions{};
    std::array<float, 20> velocities_rad_s{}, effort_a{};
    std::atomic<bool> enabled{false}, pause{false}, send_failure{false};
    // 1=缺关节，2=重复NID，3=未知NID；位置4=NaN；诊断4=未知状态、5=矛盾状态、6=未知错误码、7=流中断、8=立即停止码。
    std::atomic<int> state_fault{0}, diagnostic_fault{0};
    std::atomic<uint16_t> diagnostic_code{0}; // 语义关节8；与故障帧注入独立。
    std::atomic<int> paused_workers{0}, sends{0}, stops{0}, emergencies{0}, releases{0}, closed_subs{0};
    std::atomic<bool> emergency_keeps_enabled{true}, emergency_failure{false}, disable_failure{false};
    std::atomic<bool> clear_failure{false}, sticky_fault{false}, enable_fault{false};
    std::atomic<int> clears{0};
    // 模拟异步清错尚未完成但仍在上报新鲜Ready帧的设备。
    std::atomic<int> clear_busy_ms{0}, ignored_enables{0};
    std::atomic<uint64_t> clear_ready_ns{0};
    std::atomic<int> effort_writes{0}, mit_writes{0};
};
void enqueue(const std::shared_ptr<Scenario> &scenario);
void set_version(const char *version);
int initialization_count();
}
// 仅离线共享库入口；必须在创建任何句柄前设置。默认右手，SN按请求精确回显。
extern "C" void real_hand_test_set_default_hand(int hand);
extern "C" void real_hand_test_set_default_feedback(const float *velocities_rad_s, const float *effort_a);
