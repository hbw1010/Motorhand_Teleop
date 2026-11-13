#pragma once
#include "encos_driver.hpp"
#include "force_controller.hpp"
#include <string>
namespace motorhand::native {
// 参数在模型电机轴中计算；配置的逐轴数组以原始电机ID排列。
struct DriverImpedanceConfig {
    double stiffness_min=5, stiffness_max=60, gain=1, damping_s=.008, damping_max=.5;
    Vec12 joint_gain{1.5,1,1,1,1,1,1,1,1,1,1,1};
    Vec12 kp_max{100,80,80,80,80,80,80,80,80,80,80,80};
    double lookahead_s=.02, damping_off_velocity=.15, damping_on_velocity=.05, torque_margin=.85;
    double surface_rate=1, inverse_regularization=.01, max_proxy_offset=.15, ramp_s=.15;
    double velocity_filter_hz=12, velocity_deadband=.02, velocity_scale=.08, friction_gain=1, friction_max=.6;
    Vec12 coulomb_nm{.25,.10,.25,.25,.25,.25,.25,.25,.25,.25,.25,.25};
    double viscous=0;
    bool friction_enabled=true;
};
DriverImpedanceConfig load_driver_impedance_config(const std::string& path);
struct DriverAxisCommand { double target_rad{},kp{},kd{},feedforward_nm{}; };
// 总PD预算含阻尼与编码余量；只保证当前状态下的名义输出，MIT没有硬力矩上限。
DriverAxisCommand plan_driver_axis(double q,double dq,double surface,int direction,double stiffness,
    double torque_limit,double quantization_reserve,const DriverImpedanceConfig&,bool& damping_on,
    DriverAxisObservation* observation=nullptr);
class MotorVelocityEstimator {
public:
    explicit MotorVelocityEstimator(double cutoff_hz);
    Vec12 update(const GloveFrame&,std::uint64_t now_ns);
    void reset();
private:
    double cutoff_; Vec12 q_{},velocity_{}; std::array<std::uint64_t,12> stamps_{};
};
struct DriverImpedancePlan {
    DriverImpedanceCommand command{};
    Vec12 nominal_nm{},proxy_surface_rad{};
    DriverImpedanceObservation observation{};
    bool valid{};
};
class DriverImpedancePlanner {
public:
    DriverImpedancePlanner(DriverImpedanceConfig,double hardness,const std::array<EncosMotorCalibration,12>&);
    void reset(std::uint64_t now_ns);
    DriverImpedancePlan step(const Vec20& final_target,const HandFrame&,const GloveFrame&,const GlovePose&,
                            const std::array<double,240>& final_jacobian,const ForceResult&,std::uint64_t now_ns);
private:
    DriverImpedanceConfig config_; double hardness_;
    std::array<EncosMotorCalibration,12> calibration_; MotorVelocityEstimator velocity_;
    std::array<int,12> direction_{}; std::array<bool,12> damping_{};Vec12 surface_{};
    std::uint64_t reset_ns_{},source_ns_{};
};
} // namespace motorhand::native
