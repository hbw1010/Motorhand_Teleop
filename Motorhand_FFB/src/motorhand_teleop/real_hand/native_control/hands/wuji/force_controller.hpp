#pragma once

#include "types.hpp"
#include "common/feedback.hpp"
#include "linear_force.hpp"
#include <array>
#include <cstdint>
#include <string>

namespace motorhand::native {

struct ForceResult {
    Vec20 virtual_hand_nm{}; // Software impedance, not measured actuator torque.
    Vec20 wuji_load_fraction{}; // Current-normalized proxy, not contact force.
    Vec12 model_motor_nm{};  // Nominal model-axis torque.
    std::array<bool, 20> contact{};
    // contact含释放迟滞；只有此逐轴mask允许施加阻抗。回开/释放立即撤销。
    std::array<bool,20> impedance_contact{};
    std::array<int,20> contact_direction{};
    bool active{};           // Inputs and local geometry are valid, not a contact indication.
    bool linear{};
    ContactPhase phase{ContactPhase::not_configured};
    Vec12 projected_motor_nm{}; // 未爬升/限幅/斜率处理的模型电机目标，仅用于遥测。
    Vec20 filtered_current_a{};
    double reflection_gain{};
    std::array<double, 240> hand_to_motor{}; // 12x20行主序，未乘反射增益。
    Vec20 projection_current_a{};
    std::uint64_t projection_received_ns{};
};

class ForceController : public HandFeedback<20> {
public:
    // raw_to_model maps raw motor ID minus one to one-based CAD motor row.
    // sign and scale are in raw ID order. Install calibrated limits before use.
    ForceController(const std::string& contact_path, const std::string& linear_path,
                    const Vec12& sign, const Vec12& scale,
                    const std::array<int, 12>& raw_to_model, bool linear_mode = false);
    void set_motor_limits(const Vec12& lower_rad, const Vec12& upper_rad);
    void reset(std::uint64_t now_ns);
    ForceResult step(const Vec20& final_target, const HandFrame& hand,
                     const GloveFrame& glove, const GlovePose& pose,
                     const std::array<double, 240>& retarget_jacobian,
                     const Vec12& jac_reference_joint_q12,
                     std::uint64_t jac_source_ns, std::uint64_t now_ns);
    FeedbackResult<20> step(const Vec20&, const HandState<20>&, const GloveFrame&, const GlovePose&,
        const RetargetJacobian<20>&, const Vec12&, std::uint64_t, std::uint64_t) override;
    void acknowledge_torque(const Vec12& actual);
private:
    void clear_contact();
    void clear_axis(int axis);
    Vec20 current_on_{}, current_off_{}, error_on_{}, error_off_{}, stalled_{}, stiffness_{}, damping_{}, bias_{};
    LinearForceParameters linear_parameters_{};
    bool linear_mode_{}, linear_geometry_active_{};
    double dwell_{}, release_dwell_{}, current_hz_{}, output_hz_{}, max_gap_{}, load_scale_{}, opening_deadband_{}, velocity_limit_{}, reflection_gain_{};
    double gain_{}, ramp_{}, feedback_age_{}, jac_age_{}, jac_pose_delta_{};
    Vec12 max_torque_{}, max_ack_torque_{}, codec_zero_{}, max_slew_{}, lower_{}, upper_{};
    bool limits_set_{}, reset_set_{}, have_source_{}, have_hand_{}, have_current_{}, pending_{};
    std::uint64_t reset_ns_{}, history_ns_{}, source_ns_{}, hand_ns_{}, pending_ns_{};
    Vec20 last_target_{}, last_position_{}, filtered_current_{}, output_{};
    Vec12 history_{}, pending_output_{};
    std::array<bool, 20> active_{}, candidate_set_{}, release_set_{};
    std::array<int, 20> direction_{}, candidate_direction_{};
    std::array<std::uint64_t, 20> candidate_ns_{}, release_ns_{};
};
} // namespace motorhand::native

namespace motorhand::native {
inline FeedbackResult<20> ForceController::step(const Vec20& target,const HandState<20>& hand,const GloveFrame& glove,
    const GlovePose& pose,const RetargetJacobian<20>& jac,const Vec12& reference,std::uint64_t source,std::uint64_t now) {
    if (hand.has_load && (hand.load_unit!=HandLoadUnit::ampere || hand.load_source!="wuji-current"))
        throw std::runtime_error("Wuji反馈仅接受原始电流A");
    HandFrame legacy{hand.position_rad,hand.velocity_rad_s,hand.load,hand.has_velocity,hand.has_load,hand.received_ns};
    auto value=step(target,legacy,glove,pose,jac,reference,source,now);
    FeedbackResult<20> result;
    result.virtual_hand_nm=value.virtual_hand_nm;
    result.load_fraction=value.wuji_load_fraction;
    result.model_motor_nm=value.model_motor_nm;
    result.contact=value.contact;
    result.impedance_contact=value.impedance_contact;
    result.contact_direction=value.contact_direction;
    result.active=value.active;
    result.linear=value.linear;
    result.phase=value.phase;
    result.projected_motor_nm=value.projected_motor_nm;
    result.filtered_load=value.filtered_current_a;
    result.reflection_gain=value.reflection_gain;
    result.hand_to_motor=value.hand_to_motor;
    result.projection_load=value.projection_current_a;
    result.projection_received_ns=value.projection_received_ns;
    return result;
}
}
