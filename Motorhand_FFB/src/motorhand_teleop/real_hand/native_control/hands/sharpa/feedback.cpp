#include "feedback.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace motorhand::native::sharpa {
namespace {
using Json = nlohmann::json;
constexpr double half_lsb = 10. / 4095.;
template <class T> bool finite(const T &x) {
  for (double v : x)
    if (!std::isfinite(v))
      return false;
  return true;
}
bool fresh(std::uint64_t stamp, std::uint64_t now, double age) {
  return stamp && stamp <= now && (now - stamp) * 1e-9 < age;
}
double number(const Json &j) {
  if (!j.is_number())
    throw std::invalid_argument("Sharpa反馈参数必须为数字");
  double x = j.get<double>();
  if (!std::isfinite(x))
    throw std::invalid_argument("Sharpa反馈参数必须有限");
  return x;
}
double nonnegative(const Json &j, const char *key) {
  double x = number(j.at(key));
  if (x < 0)
    throw std::invalid_argument(std::string("Sharpa反馈参数不能为负: ") + key);
  return x;
}
template <std::size_t N>
std::array<double, N> array(const Json &j, const char *key) {
  const auto &a = j.at(key);
  if (!a.is_array() || a.size() != N)
    throw std::invalid_argument(std::string("Sharpa反馈参数维度错误: ") + key);
  std::array<double, N> out;
  for (std::size_t i = 0; i < N; ++i)
    out[i] = number(a[i]);
  return out;
}
} // namespace
Feedback::Feedback(const std::string &path, bool linear,
                   const std::array<EncosMotorCalibration, 12> &cal,
                   std::string expected_source)
    : expected_source_(std::move(expected_source)), linear_(linear) {
  if (expected_source_ != torque_source &&
      expected_source_ != "sharpa-mujoco-contact-joint-torque")
    throw std::invalid_argument("Sharpa反馈负载来源不受支持");
  std::ifstream file(path);
  if (!file)
    throw std::invalid_argument("无法读取Sharpa反馈配置: " + path);
  Json j;
  file >> j;
  if (j.at("schema_version") != 1 || j.at("load_unit") != "newton_metre" ||
      j.at("load_source") != torque_source ||
      j.at("load_reference_source") != "urdf-effort-nominal-normalization" ||
      j.at("estimate_acknowledged") != true)
    throw std::invalid_argument("Sharpa反馈必须明确SDK关节力矩来源与名义估计");
  reference_ = array<22>(j, "joint_load_reference_nm");
  bias_ = array<22>(j, "joint_torque_bias_nm");
  sign_ = array<22>(j, "joint_torque_sign");
  on_ = array<22>(j, "contact_on_nm");
  off_ = array<22>(j, "contact_off_nm");
  error_on_ = array<22>(j, "error_on_rad");
  error_off_ = array<22>(j, "error_off_rad");
  stalled_ = array<22>(j, "stalled_speed_rad_s");
  stiffness_ = array<22>(j, "stiffness_nm_per_rad");
  damping_ = array<22>(j, "damping_nm_s_per_rad");
  for (int i = 0; i < 22; ++i)
    if (reference_[i] <= 0 || std::abs(sign_[i]) != 1 || on_[i] <= off_[i] ||
        off_[i] < 0 || error_on_[i] <= error_off_[i] || error_off_[i] < 0 ||
        stalled_[i] <= 0 || stiffness_[i] <= 0 || damping_[i] < 0)
      throw std::invalid_argument("Sharpa力矩/接触逐轴参数无效");
  gain_ = nonnegative(j, "gain");
  reflection_ = nonnegative(j, "reflection_gain");
  ramp_ = nonnegative(j, "ramp_duration_s");
  dwell_ = nonnegative(j, "dwell_s");
  release_dwell_ = nonnegative(j, "release_dwell_s");
  filter_hz_ = nonnegative(j, "torque_filter_hz");
  feedback_age_ = nonnegative(j, "feedback_max_age_s");
  jac_age_ = nonnegative(j, "jacobian_max_age_s");
  jac_pose_delta_ = nonnegative(j, "jacobian_max_pose_delta_rad");
  opening_deadband_ = nonnegative(j, "opening_speed_deadband_rad_s");
  if (gain_ <= 0 || gain_ > 1 || reflection_ <= 0 || reflection_ > 10 ||
      ramp_ <= 0 || dwell_ <= 0 || release_dwell_ <= 0 || filter_hz_ <= 0 ||
      feedback_age_ <= 0 || feedback_age_ > .1 || jac_age_ <= 0 ||
      jac_pose_delta_ <= 0)
    throw std::invalid_argument("Sharpa反馈公共参数无效");
  const auto raw_cap = array<12>(j, "glove_max_output_torque_nm"),
             raw_slew = array<12>(j, "max_torque_slew_nm_s");
  std::array<bool, 12> used{};
  for (int model = 0; model < 12; ++model) {
    const auto &c = cal[model];
    int raw = c.raw_motor_id - 1;
    double factor = std::abs(c.sign * c.scale);
    if (raw < 0 || raw >= 12 || used[raw] || !std::isfinite(factor) ||
        factor <= 0 || !c.has_torque_limit ||
        !std::isfinite(c.raw_torque_limit_nm) || raw_cap[raw] <= half_lsb ||
        raw_cap[raw] > 10 || raw_cap[raw] > c.raw_torque_limit_nm + 1e-12 ||
        raw_slew[raw] <= 0)
      throw std::invalid_argument("Sharpa反馈Encos绑定/限额无效");
    used[raw] = true;
    cap_[model] = std::nextafter(raw_cap[raw] - half_lsb, 0.) / factor;
    ack_cap_[model] = raw_cap[raw] / factor;
    slew_[model] = raw_slew[raw] / factor;
    if (!std::isfinite(cap_[model]) || !std::isfinite(ack_cap_[model]) ||
        !std::isfinite(slew_[model]))
      throw std::invalid_argument("Sharpa反馈传动比例产生非法限额");
  }
}
void Feedback::set_motor_limits(const Vec12 &lo, const Vec12 &hi) {
  if (!finite(lo) || !finite(hi))
    throw std::invalid_argument("Sharpa反馈电机限位必须有限");
  for (int i = 0; i < 12; ++i)
    if (lo[i] >= hi[i])
      throw std::invalid_argument("Sharpa反馈电机限位无效");
  lower_ = lo;
  upper_ = hi;
  limits_set_ = true;
}
void Feedback::clear_contact() {
  contact_ = {};
  candidate_ = {};
  releasing_ = {};
  direction_ = {};
  candidate_direction_ = {};
  candidate_ns_ = {};
  release_ns_ = {};
}
void Feedback::reset(std::uint64_t now) {
  clear_contact();
  history_ = {};
  filtered_ = {};
  have_source_ = have_hand_ = pending_ = geometry_active_ = false;
  reset_set_ = true;
  reset_ns_ = history_ns_ = now;
  source_ns_ = hand_ns_ = 0;
}
FeedbackResult<22>
Feedback::step(const JointArray<22> &target, const HandState<22> &hand,
               const GloveFrame &glove, const GlovePose &pose,
               const RetargetJacobian<22> &jac, const Vec12 &jac_reference,
               std::uint64_t jac_ns, std::uint64_t now) {
  auto stop = [&]() {
    clear_contact();
    pending_ = false;
    throw std::runtime_error("Sharpa力反馈缺少新鲜完整SDK关节力矩Nm或有效几何");
  };
  if (!reset_set_ || !limits_set_ || now < reset_ns_ || now < history_ns_ ||
      !hand.connected || hand.fault || !hand.has_load ||
      hand.load_unit != HandLoadUnit::newton_metre ||
      hand.load_source != expected_source_ ||
      !fresh(hand.received_ns, now, feedback_age_) ||
      !fresh(glove.oldest_ns, now, feedback_age_) || !finite(target) ||
      !finite(hand.position_rad) || !finite(hand.load) ||
      (hand.has_velocity && !finite(hand.velocity_rad_s)) ||
      !finite(glove.motor_rad) || !finite(pose.motor_rad) ||
      !finite(pose.joint_rad) || !finite(pose.motor_jacobian) || !finite(jac) ||
      !finite(jac_reference))
    stop();
  bool geometry = fresh(jac_ns, now, jac_age_) && jac_ns <= glove.oldest_ns;
  std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
  for (int i = 0; i < 12; ++i) {
    if (!fresh(glove.received_ns[i], now, feedback_age_) ||
        std::abs(pose.motor_rad[i] - glove.motor_rad[i]) > 1e-7)
      stop();
    if (std::abs(pose.joint_rad[i] - jac_reference[i]) > jac_pose_delta_)
      geometry = false;
    oldest = std::min(oldest, glove.received_ns[i]);
  }
  if (oldest != glove.oldest_ns)
    stop();
  const auto source = std::min(hand.received_ns, glove.oldest_ns);
  if ((have_source_ && source < source_ns_) ||
      (have_hand_ && hand.received_ns < hand_ns_))
    stop();
  using Mat12 = Eigen::Matrix<double, 12, 12, Eigen::RowMajor>;
  using Mat22 = Eigen::Matrix<double, 22, 12, Eigen::RowMajor>;
  using Projection = Eigen::Matrix<double, 12, 22, Eigen::RowMajor>;
  const Eigen::Map<const Mat12> motor(pose.motor_jacobian.data());
  const Eigen::Map<const Mat22> retarget(jac.data());
  Eigen::FullPivLU<Mat12> lu(motor.transpose());
  if (!lu.isInvertible() || lu.rcond() < 1e-8)
    stop();
  const Projection projection = lu.solve(retarget.transpose());
  if (!projection.allFinite() || retarget.norm() < 1e-8)
    geometry = false;
  FeedbackResult<22> out;
  out.linear = linear_;
  out.active = geometry;
  out.reflection_gain = gain_ * reflection_;
  out.projection_load = hand.load;
  out.projection_received_ns = hand.received_ns;
  std::copy_n(projection.data(), 264, out.hand_to_motor.begin());
  const bool new_hand = !have_hand_ || hand.received_ns > hand_ns_,
             new_frame = !have_source_ || source > source_ns_;
  const double dt = have_source_ ? (source - source_ns_) * 1e-9 : 0,
               hand_dt = have_hand_ ? (hand.received_ns - hand_ns_) * 1e-9 : 0;
  JointArray<22> velocity = hand.velocity_rad_s;
  bool has_velocity = hand.has_velocity;
  if (!has_velocity && have_hand_ && hand_dt > 0) {
    has_velocity = true;
    for (int i = 0; i < 22; ++i)
      velocity[i] = (hand.position_rad[i] - last_position_[i]) / hand_dt;
  }
  if (!geometry)
    clear_contact();
  for (int i = 0; i < 22; ++i) {
    if (!hand.online[i] || !hand.enabled[i])
      stop();
    double corrected = (hand.load[i] - bias_[i]) * sign_[i];
    if (!std::isfinite(corrected))
      stop();
    if (new_hand) {
      double alpha =
          have_hand_ ? -std::expm1(-6.283185307179586 * filter_hz_ * hand_dt)
                     : 1;
      filtered_[i] += alpha * (corrected - filtered_[i]);
    }
    out.filtered_load[i] = filtered_[i];
    out.load_fraction[i] = std::clamp(corrected / reference_[i], -1., 1.);
    if (linear_ || !geometry)
      continue;
    const double error = target[i] - hand.position_rad[i];
    const int dir = (error > 0) - (error < 0);
    if (!std::isfinite(error))
      stop();
    const bool observable = projection.col(i).norm() > 1e-8;
    double speed = has_velocity ? velocity[i] : 0;
    if (!observable) {
      contact_[i] = candidate_[i] = releasing_[i] = false;
      direction_[i] = candidate_direction_[i] = 0;
      continue;
    }
    bool release = false;
    if (contact_[i]) {
      release = direction_[i] * error <= error_off_[i] ||
                std::abs(filtered_[i]) <= off_[i];
      if (release) {
        if (!releasing_[i]) {
          releasing_[i] = true;
          release_ns_[i] = source;
        }
        if ((source - release_ns_[i]) * 1e-9 >= release_dwell_) {
          contact_[i] = false;
          direction_[i] = 0;
        }
      } else
        releasing_[i] = false;
    }
    if (!contact_[i] && new_frame) {
      bool blocked = std::abs(error) >= error_on_[i] &&
                     std::abs(filtered_[i]) >= on_[i] && has_velocity &&
                     std::abs(speed) <= stalled_[i];
      if (!blocked) {
        candidate_[i] = false;
        candidate_direction_[i] = 0;
      } else {
        if (!candidate_[i] || candidate_direction_[i] != dir) {
          candidate_[i] = true;
          candidate_direction_[i] = dir;
          candidate_ns_[i] = source;
        }
        if ((source - candidate_ns_[i]) * 1e-9 >= dwell_) {
          contact_[i] = true;
          direction_[i] = dir;
          candidate_[i] = false;
        }
      }
    }
    const bool opening =
        have_source_ && direction_[i] * (target[i] - last_target_[i]) <
                            -(opening_deadband_ * std::max(dt, 1e-3));
    if (contact_[i] && !release && !releasing_[i] && !opening) {
      const int n = direction_[i];
      double penetration = std::max(0., n * error - error_off_[i]);
      double target_velocity = dt > 0 ? (target[i] - last_target_[i]) / dt : 0;
      double relative = n * (target_velocity - speed);
      if (relative >= -opening_deadband_) {
        out.virtual_hand_nm[i] =
            -n *
            std::min(reference_[i], std::max(0., stiffness_[i] * penetration +
                                                     damping_[i] * relative));
        out.impedance_contact[i] = n * out.virtual_hand_nm[i] < 0;
      }
    }
    out.contact[i] = contact_[i];
    out.contact_direction[i] = direction_[i];
  }
  if (new_hand) {
    hand_ns_ = hand.received_ns;
    last_position_ = hand.position_rad;
    have_hand_ = true;
  }
  if (new_frame) {
    source_ns_ = source;
    last_target_ = target;
    have_source_ = true;
  }
  if (linear_ && geometry && !geometry_active_)
    reset_ns_ = now;
  geometry_active_ = geometry;
  const double x = std::clamp((now - reset_ns_) * 1e-9 / ramp_, 0., 1.),
               ramp = x * x * x * (10 + x * (-15 + 6 * x)),
               history_dt = (now - history_ns_) * 1e-9;
  for (int row = 0; row < 12; ++row) {
    double projected = 0;
    if (geometry) {
      if (linear_) {
        double capacity = 0;
        for (int col = 0; col < 22; ++col) {
          capacity += std::abs(projection(row, col));
          projected += projection(row, col) * out.load_fraction[col];
        }
        projected = capacity > 0 ? -gain_ * cap_[row] *
                                       std::clamp(projected / capacity, -1., 1.)
                                 : 0;
      } else
        for (int col = 0; col < 22; ++col)
          projected += gain_ * reflection_ * projection(row, col) *
                       out.virtual_hand_nm[col];
    }
    if (!std::isfinite(projected))
      stop();
    out.projected_motor_nm[row] = projected;
    double wanted = std::clamp(projected * ramp, -cap_[row], cap_[row]);
    if (!geometry)
      wanted = 0;
    else if (linear_)
      wanted = std::clamp(wanted, history_[row] - slew_[row] * history_dt,
                          history_[row] + slew_[row] * history_dt);
    else if (history_[row] * wanted < 0)
      wanted = 0;
    else
      wanted =
          std::copysign(std::min(std::abs(wanted), std::abs(history_[row]) +
                                                       slew_[row] * history_dt),
                        wanted);
    if ((pose.motor_rad[row] <= lower_[row] && wanted < 0) ||
        (pose.motor_rad[row] >= upper_[row] && wanted > 0))
      wanted = 0;
    out.model_motor_nm[row] = wanted;
  }
  if (linear_)
    out.phase = ContactPhase::not_configured;
  else if (!geometry)
    out.phase = ContactPhase::waiting_geometry;
  else {
    bool contact = false, release = false, candidate = false;
    for (int i = 0; i < 22; ++i) {
      contact |= contact_[i] && !releasing_[i];
      release |= releasing_[i];
      candidate |= candidate_[i];
    }
    out.phase = contact     ? ContactPhase::contact
                : release   ? ContactPhase::releasing
                : candidate ? ContactPhase::arming
                            : ContactPhase::free;
  }
  pending_output_ = out.model_motor_nm;
  pending_ns_ = now;
  pending_ = true;
  return out;
}
void Feedback::acknowledge_torque(const Vec12 &actual) {
  if (!pending_ || !finite(actual))
    throw std::runtime_error("Sharpa反馈力矩确认无效");
  for (int i = 0; i < 12; ++i)
    if (std::abs(actual[i]) > ack_cap_[i] + 1e-12)
      throw std::runtime_error("Sharpa反馈确认超过名义限额");
  for (int i = 0; i < 12; ++i) {
    double zero = ack_cap_[i] - cap_[i];
    bool zero_bin = std::abs(std::abs(actual[i]) - zero) < 1e-12 + 1e-9 * zero;
    history_[i] = zero_bin && pending_output_[i] * actual[i] <= 0
                      ? 0
                      : std::clamp(actual[i], -cap_[i], cap_[i]);
  }
  history_ns_ = pending_ns_;
  pending_ = false;
}
} // namespace motorhand::native::sharpa
