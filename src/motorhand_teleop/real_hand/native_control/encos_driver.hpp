#pragma once

#include "types.hpp"
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
    void connect();
    GloveFrame read();
    std::array<EncosMotorCalibration, 12> calibration() const;
    // Monotonic timestamps after serial submission, indexed in GLOVE_JOINTS order.
    std::array<std::uint64_t, 12> last_command_write_ns() const;
    void home(double tolerance_rad = 0.15, double timeout_s = 45.0);
    void enable_force();
    Vec12 write_position(const Vec12& motor_rad, std::uint64_t deadline_ns);
    Vec12 write_torque(const Vec12& nominal_nm, std::uint64_t deadline_ns);
    // If output was acquired, disable verifies all 12 zero-current serial
    // submissions within 100 ms; failure is reported and output stays latched.
    // Serial submission is not motor acknowledgement or measured torque.
    void disable();
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace motorhand::native
