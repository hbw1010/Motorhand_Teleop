#pragma once
#include "common/glove_device.hpp"
namespace motorhand::native::simulation {
// 20ms独立采样的无硬件手套。输出仅保存名义值，不模拟人手受力动力学。
class FakeGlove final : public GloveDevice {
public:
    explicit FakeGlove(std::array<EncosMotorCalibration,12> calibration);
    ~FakeGlove() override;
    FakeGlove(const FakeGlove&)=delete;
    FakeGlove& operator=(const FakeGlove&)=delete;
    void connect() override;
    GloveFrame read() override;
    std::array<EncosMotorCalibration,12> calibration() const override;
    std::array<std::uint64_t,12> last_command_write_ns() const override;
    void home() override;
    void enable_force() override;
    Vec12 write_position(const Vec12&,std::uint64_t original_deadline_ns) override;
    Vec12 write_torque(const Vec12&,std::uint64_t original_deadline_ns) override;
    DriverImpedanceSubmission write_impedance(const DriverImpedanceCommand&,std::uint64_t original_deadline_ns) override;
    void disable() override;
    void close() override;
    void set_manual_target(const Vec12& motor_rad);
    Vec12 last_nominal_nm() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
