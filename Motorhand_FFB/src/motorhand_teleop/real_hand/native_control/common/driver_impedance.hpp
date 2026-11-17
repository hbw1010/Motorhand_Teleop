#pragma once
#include "common/driver_impedance_config.hpp"
#include "common/feedback.hpp"
namespace motorhand::native {
template<std::size_t N> struct CommonDriverImpedancePlan {
    DriverImpedanceCommand command{};
    Vec12 nominal_nm{},proxy_surface_rad{};
    CommonDriverImpedanceObservation<N> observation{};
    bool valid{};
};
template<std::size_t N> class CommonDriverImpedancePlanner {
public:
    CommonDriverImpedancePlanner(DriverImpedanceConfig,double hardness,const std::array<EncosMotorCalibration,12>&);
    void reset(std::uint64_t now_ns);
    CommonDriverImpedancePlan<N> step(const JointArray<N>& final_target,const HandState<N>&,const GloveFrame&,const GlovePose&,
                            const RetargetJacobian<N>& final_jacobian,const FeedbackResult<N>&,std::uint64_t now_ns);
private:
    DriverImpedanceConfig config_; double hardness_;
    std::array<EncosMotorCalibration,12> calibration_; MotorVelocityEstimator velocity_;
    std::array<int,12> direction_{}; std::array<bool,12> damping_{};Vec12 surface_{};
    std::uint64_t reset_ns_{},source_ns_{};
};
}
