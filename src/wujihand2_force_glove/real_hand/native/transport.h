#pragma once
#include "real_hand.h"
#include <array>
#include <chrono>
#include <memory>
#include <string>

namespace real_hand {
using Angles = std::array<double, 20>;
inline uint64_t monotonic_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Identity {
    int model = 0;
    int hand = -1;
    std::string serial;
    std::string firmware;
    Angles lower{}, upper{};
};
struct Feedback {
    Angles positions{};
    // 与positions同帧、同NID语义顺序；effort单位A，不使用诊断current替代。
    Angles velocities_rad_s{}, effort_a{};
    bool has_velocity = false, has_effort = false;
    std::array<uint8_t, 20> enabled{}, online{};
    std::array<uint32_t, 20> errors{};
    // 仅固定SDK目录确认severity精确为Warning；errors始终保留原始码。
    std::array<bool, 20> warnings{};
    uint64_t position_ns = 0, diagnostics_ns = 0, device_us = 0;
    bool complete = false;
};
// 只在控制互斥锁下调用；适配器的反馈回调使用独立短锁。
// release 必须结束回调并释放本手全部句柄，即使断开失败也不能遗漏释放。
class Transport {
public:
    virtual ~Transport() = default;
    virtual Identity connect(const RealHandOptions &) = 0;
    virtual Feedback feedback() = 0;
    virtual void prepare(const RealHandOptions &) = 0;
    virtual void send(const Angles &) = 0;
    virtual void enable() = 0;
    virtual void stop(bool emergency) = 0;
    virtual void clear_fault() = 0;
    virtual void release() = 0;
};
std::unique_ptr<Transport> make_transport();
}
