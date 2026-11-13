#pragma once

#include "types.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace motorhand::native {

// Indexed in GLOVE_JOINTS order; torque caps are nominal, not measured output.
struct EncosMotorCalibration {
    int raw_motor_id{};
    double zero_deg{};
    double sign{};
    double scale{};
    double raw_min_deg{};
    double raw_max_deg{};
    double model_min_rad{};
    double model_max_rad{};
    double raw_torque_limit_nm{};
    bool has_torque_limit{};
};

class EncosDriver {
public:
    explicit EncosDriver(const std::string& mapping_path,
                         const std::string& force_limits_path = {});
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
    // Explicit PTY-only injection, unavailable in production builds.
    struct OfflineTestPorts { std::array<std::string, 3> paths; };
    EncosDriver(const std::string& mapping_path, const std::string& force_limits_path,
                const OfflineTestPorts& ports);
#endif
    ~EncosDriver();
    EncosDriver(const EncosDriver&) = delete;
    EncosDriver& operator=(const EncosDriver&) = delete;

    // Connection/read-only polling sends the SDK's zero-current servo probes;
    // this is not passive bus listening.
    // Transport: MOTORHAND_GLOVE_DRIVER=1 (3x CANable2 SLCAN) or 2 (STM32G4 LQS-GW02
    // gateway via the vendor HandClientG4 SDK, port auto or MOTORHAND_GLOVE_PORT);
    // unset picks whichever hardware is plugged in.  Everything above the wire is shared.
    void connect();
    GloveFrame read();
    std::array<EncosMotorCalibration, 12> calibration() const;
    // Monotonic timestamps after serial submission, indexed in GLOVE_JOINTS order.
    std::array<std::uint64_t, 12> last_command_write_ns() const;
    void home(double tolerance_rad = 0.15, double timeout_s = 45.0);
    // Treat the glove as homed without moving it: it was homed earlier (a reconnect after a motor fault resumes
    // from where the hand is instead of dragging it back to zero). Same checks as home(), no output acquired.
    void mark_homed();
    void enable_force();
    Vec12 write_position(const Vec12& motor_rad, std::uint64_t deadline_ns);
    Vec12 write_torque(const Vec12& nominal_nm, std::uint64_t deadline_ns);
    // 量化后按当前反馈位置/输入估算速度检查总PD预算及撤离方向。
    // MIT没有硬力矩限额；返回值不是实测力矩，后续姿态变化会改变PD。
    DriverImpedanceSubmission write_impedance(const DriverImpedanceCommand&, std::uint64_t deadline_ns);
    // If output was acquired, disable verifies all 12 zero-current serial
    // submissions within 100 ms; failure is reported and output stays latched.
    // Serial submission is not motor acknowledgement or measured torque.
    void disable();
    void close();

    // Settings, defaults unchanged. slot (驱动 1 only; 驱动 2 runs at the SDK's 200 Hz): how long each adapter's thread spends on one motor before asking the
    // next; with 4 motors an adapter, each is read every 4 x slot (5 ms: 20 ms, readings ~25 ms old; 1 ms: ~4.7 ms).
    // position_kp: write_position's MIT stiffness, Nm/rad (45: a motor held 9 deg off pushes ~7 Nm all the time).
    void set_slot(std::chrono::microseconds slot);
    std::chrono::microseconds slot() const;
    void set_position_kp(double kp_nm_per_rad);
    double position_kp() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace motorhand::native
