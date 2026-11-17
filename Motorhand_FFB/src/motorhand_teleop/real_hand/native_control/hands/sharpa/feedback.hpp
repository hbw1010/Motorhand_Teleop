#pragma once
#include "common/feedback.hpp"
#include "encos_driver.hpp"
#include "identity.hpp"
namespace motorhand::native::sharpa {
// 仅使用SDK报告关节力矩；偏置/符号是反馈参数，绝不写入设备。
class Feedback final : public HandFeedback<22> {
public:
  Feedback(const std::string &config_path, bool linear_mode,
           const std::array<EncosMotorCalibration, 12> &,
           std::string expected_source = torque_source);
  void set_motor_limits(const Vec12 &, const Vec12 &) override;
  void reset(std::uint64_t now_ns) override;
  FeedbackResult<22> step(const JointArray<22> &, const HandState<22> &,
                          const GloveFrame &, const GlovePose &,
                          const RetargetJacobian<22> &, const Vec12 &,
                          std::uint64_t, std::uint64_t) override;
  void acknowledge_torque(const Vec12 &) override;

private:
  void clear_contact();
  JointArray<22> reference_{}, bias_{}, sign_{}, on_{}, off_{}, error_on_{},
      error_off_{}, stalled_{}, stiffness_{}, damping_{};
  JointArray<22> filtered_{}, last_position_{}, last_target_{};
  std::array<bool, 22> contact_{}, candidate_{}, releasing_{};
  std::array<int, 22> direction_{}, candidate_direction_{};
  std::array<std::uint64_t, 22> candidate_ns_{}, release_ns_{};
  Vec12 cap_{}, ack_cap_{}, slew_{}, lower_{}, upper_{}, history_{},
      pending_output_{};
  double gain_{}, reflection_{}, ramp_{}, dwell_{}, release_dwell_{},
      filter_hz_{}, feedback_age_{}, jac_age_{}, jac_pose_delta_{},
      opening_deadband_{};
  std::string expected_source_;
  bool linear_{}, limits_set_{}, reset_set_{}, have_source_{}, have_hand_{},
      pending_{}, geometry_active_{};
  std::uint64_t reset_ns_{}, source_ns_{}, hand_ns_{}, history_ns_{},
      pending_ns_{};
};
} // namespace motorhand::native::sharpa
