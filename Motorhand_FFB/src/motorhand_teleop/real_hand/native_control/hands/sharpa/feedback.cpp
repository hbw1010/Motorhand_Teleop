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
constexpr double full_turn = 6.283185307179586;
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
Json read_profile(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::invalid_argument("无法读取反馈配置: " + path.string());
  return Json::parse(file);
}
JointArray<22> wuji_joint_values(const Json &profile, const char *key) {
  const auto values = array<20>(profile, key);
  // 两个额外轴沿用同指近端参数。其是否参与反馈仍由Sharpa实际雅可比决定。
  constexpr std::array<int, 22> wuji_axis{
      0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19};
  JointArray<22> result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = values[wuji_axis[i]];
  return result;
}
} // namespace
Feedback::Feedback(const std::filesystem::path &path,
                   const std::filesystem::path &wuji_config_dir, bool linear,
                   const std::array<EncosMotorCalibration, 12> &cal,
                   std::string expected_source, std::uint64_t startup_delay_ns)
    : expected_source_(std::move(expected_source)), linear_(linear), startup_delay_ns_(startup_delay_ns) {
  if (expected_source_ != torque_source &&
      expected_source_ != "sharpa-mujoco-contact-joint-torque")
    throw std::invalid_argument("Sharpa反馈负载来源不受支持");
  const auto j = read_profile(path);
  if (j.at("schema_version") != 1 || j.at("load_unit") != "newton_metre" ||
      j.at("load_source") != torque_source ||
      j.at("load_reference_source") != "urdf-effort-nominal-normalization" ||
      j.at("estimate_acknowledged") != true)
    throw std::invalid_argument("Sharpa反馈必须明确SDK关节力矩来源与名义估计");
  reference_ = array<22>(j, "joint_load_reference_nm");
  bias_ = array<22>(j, "joint_torque_bias_nm");
  sign_ = array<22>(j, "joint_torque_sign");
  for (int i = 0; i < 22; ++i)
    if (reference_[i] <= 0 || std::abs(sign_[i]) != 1)
      throw std::invalid_argument("Sharpa关节力矩参考值或符号无效");
  const auto motor = linear_ ? j : read_profile(wuji_config_dir / "wuji-max-torque-linear-right.json");
  if (!linear_) {
    const auto contact = read_profile(wuji_config_dir / "wuji-contact-impedance-right.json");
    const auto current_reference = wuji_joint_values(motor, "wuji_max_current_a");
    const auto current_on = wuji_joint_values(contact, "current_on_a");
    const auto current_off = wuji_joint_values(contact, "current_off_a");
    const double load_weight_scale = nonnegative(contact, "load_weight_scale_a");
    error_on_ = wuji_joint_values(contact, "error_on_rad");
    error_off_ = wuji_joint_values(contact, "error_off_rad");
    stalled_ = wuji_joint_values(contact, "stalled_speed_rad_s");
    stiffness_ = wuji_joint_values(contact, "stiffness_nm_per_rad");
    damping_ = wuji_joint_values(contact, "damping_nm_s_per_rad");
    for (int i = 0; i < 22; ++i) {
      if (current_reference[i] <= 0 || current_on[i] <= current_off[i] ||
          current_off[i] < 0 || load_weight_scale <= 0 ||
          error_on_[i] <= error_off_[i] || error_off_[i] < 0 ||
          stalled_[i] <= 0 || stiffness_[i] <= 0 || damping_[i] < 0)
        throw std::invalid_argument("Wuji接触配置无法适配Sharpa");
      const double load_reference_ratio = reference_[i] / current_reference[i];
      on_[i] = current_on[i] * load_reference_ratio;
      off_[i] = current_off[i] * load_reference_ratio;
      load_scale_[i] = load_weight_scale * load_reference_ratio;
      if (!std::isfinite(on_[i]) || !std::isfinite(off_[i]) ||
          !std::isfinite(load_scale_[i]) || load_scale_[i] <= 0)
        throw std::invalid_argument("Wuji接触负载比例无法表示为Sharpa力矩");
    }
    dwell_ = nonnegative(contact, "dwell_s");
    release_dwell_ = nonnegative(contact, "release_dwell_s");
    filter_hz_ = nonnegative(contact, "current_filter_hz");
    output_filter_hz_ = nonnegative(contact, "output_filter_hz");
    max_gap_ = nonnegative(contact, "max_sample_gap_s");
    opening_deadband_ = nonnegative(contact, "opening_speed_deadband_rad_s");
    relative_speed_limit_ = nonnegative(contact, "relative_velocity_limit_rad_s");
    if (dwell_ <= 0 || release_dwell_ <= 0 || output_filter_hz_ <= 0 ||
        max_gap_ <= 0 || opening_deadband_ >= relative_speed_limit_)
      throw std::invalid_argument("Wuji接触时间或速度参数无效");
  } else {
    filter_hz_ = nonnegative(j, "torque_filter_hz");
  }
  gain_ = nonnegative(motor, "gain");
  reflection_ = nonnegative(j, "reflection_gain");
  ramp_ = nonnegative(motor, "ramp_duration_s");
  feedback_age_ = nonnegative(motor, "feedback_max_age_s");
  jac_age_ = nonnegative(motor, "jacobian_max_age_s");
  jac_pose_delta_ = nonnegative(motor, "jacobian_max_pose_delta_rad");
  if (gain_ <= 0 || gain_ > 1 || reflection_ <= 0 || reflection_ > 10 ||
      ramp_ <= 0 || filter_hz_ <= 0 ||
      feedback_age_ <= 0 || feedback_age_ > .1 || jac_age_ <= 0 ||
      jac_pose_delta_ <= 0)
    throw std::invalid_argument("Sharpa反馈公共参数无效");
  const auto raw_cap = array<12>(j, "glove_max_output_torque_nm"),
             raw_slew = array<12>(motor, "max_torque_slew_nm_s");
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
  for (int i = 0; i < 22; ++i)
    clear_axis(i);
}
void Feedback::clear_axis(int i) {
  contact_[i] = candidate_[i] = releasing_[i] = false;
  direction_[i] = candidate_direction_[i] = 0;
  contact_output_[i] = 0;
}
void Feedback::reset(std::uint64_t now) {
  clear_contact();
  history_ = {};
  filtered_ = {};
  have_source_ = have_hand_ = pending_ = geometry_active_ = false;
  reset_set_ = true;
  reset_ns_ = history_ns_ = now;
  mode_started_ns_ = now;
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
  // Sharpa绝对姿态启动过渡未完成时卸力，避免把未缩放Jacobian用于混合位置目标。
  bool geometry = fresh(jac_ns, now, jac_age_) && jac_ns <= glove.oldest_ns &&
                  now >= mode_started_ns_ && now - mode_started_ns_ >= startup_delay_ns_;
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
  if (!linear_ && have_source_ && (source - source_ns_) * 1e-9 > max_gap_) {
    clear_contact();
    have_source_ = have_hand_ = false;
  }
  const bool new_hand = !have_hand_ || hand.received_ns > hand_ns_,
             new_frame = !have_source_ || source > source_ns_;
  const double dt = have_source_ ? (source - source_ns_) * 1e-9 : 0,
               hand_dt = have_hand_ ? (hand.received_ns - hand_ns_) * 1e-9 : 0;
  JointArray<22> velocity = hand.velocity_rad_s;
  bool has_velocity = hand.has_velocity;
  if (!has_velocity && new_frame && have_hand_ && hand_dt > 0) {
    has_velocity = true;
    for (int i = 0; i < 22; ++i)
      velocity[i] = (hand.position_rad[i] - last_position_[i]) / hand_dt;
  }
  if (!geometry)
    clear_contact();
  const double load_alpha = -std::expm1(-full_turn * filter_hz_ * (linear_ ? hand_dt : dt));
  const double output_alpha = -std::expm1(-full_turn * output_filter_hz_ * dt);
  for (int i = 0; i < 22; ++i) {
    if (!hand.online[i] || !hand.enabled[i])
      stop();
    double corrected = (hand.load[i] - bias_[i]) * sign_[i];
    if (!std::isfinite(corrected))
      stop();
    if (linear_ ? new_hand : new_frame) {
      const double sample = linear_ ? corrected : std::abs(corrected);
      const bool initialized = linear_ ? have_hand_ : have_source_;
      filtered_[i] = initialized ? filtered_[i] + load_alpha * (sample - filtered_[i]) : sample;
    }
    out.filtered_load[i] = filtered_[i];
    out.load_fraction[i] = std::clamp(corrected / reference_[i], -1., 1.);
    if (linear_ || !geometry)
      continue;
    const double error = target[i] - hand.position_rad[i];
    const int dir = (error > 0) - (error < 0);
    if (!std::isfinite(error) || !std::isfinite(filtered_[i]) ||
        (has_velocity && !std::isfinite(velocity[i])))
      stop();
    if (retarget.row(i).isZero(0)) {
      clear_axis(i);
      continue;
    }
    const double speed = has_velocity ? velocity[i] : 0;
    if (contact_[i]) {
      if (direction_[i] * error <= error_off_[i] || filtered_[i] <= off_[i]) {
        contact_output_[i] = 0;
        if (!releasing_[i]) {
          releasing_[i] = true;
          release_ns_[i] = source;
        }
        if ((source - release_ns_[i]) * 1e-9 >= release_dwell_)
          clear_axis(i);
        continue;
      }
      releasing_[i] = false;
    } else {
      if (!new_frame) {
        if (std::abs(error) < error_on_[i] || dir != candidate_direction_[i] ||
            (has_velocity && std::abs(speed) > stalled_[i]))
          clear_axis(i);
        continue;
      }
      if (std::abs(error) < error_on_[i] || filtered_[i] < on_[i] ||
          !has_velocity || std::abs(speed) > stalled_[i]) {
        clear_axis(i);
        continue;
      }
      if (!candidate_[i] || candidate_direction_[i] != dir) {
        candidate_[i] = true;
        candidate_direction_[i] = dir;
        candidate_ns_[i] = source;
      }
      if ((source - candidate_ns_[i]) * 1e-9 < dwell_)
        continue;
      contact_[i] = true;
      direction_[i] = dir;
      candidate_[i] = false;
      candidate_direction_[i] = 0;
    }
    if (!new_frame) {
      if (have_source_ && direction_[i] * (target[i] - last_target_[i]) < 0)
        contact_output_[i] = 0;
      continue;
    }
    const int n = direction_[i];
    const double penetration = std::max(0., n * error - error_off_[i]);
    double relative = 0;
    if (have_source_ && dt > 0 && has_velocity)
      relative = std::clamp(n * ((target[i] - last_target_[i]) / dt - speed),
                            -relative_speed_limit_, relative_speed_limit_);
    if (relative < -opening_deadband_) {
      contact_output_[i] = 0;
      continue;
    }
    const double load = std::min(filtered_[i] / load_scale_[i], 20.0);
    const double speed_fraction = std::min(std::abs(relative) / stalled_[i], 20.0);
    const double load_weight = -std::expm1(-load * load);
    const double damping_weight = .6 / (1 + std::exp(-load)) + .4 / (1 + std::exp(-speed_fraction));
    const double requested = -n * std::max(0., load_weight * stiffness_[i] * penetration +
                                               damping_weight * damping_[i] * relative);
    contact_output_[i] += output_alpha * (requested - contact_output_[i]);
    if (!std::isfinite(contact_output_[i]))
      stop();
  }
  if (!linear_) {
    out.virtual_hand_nm = contact_output_;
    out.contact = contact_;
    out.contact_direction = direction_;
    for (int i = 0; i < 22; ++i)
      out.impedance_contact[i] = geometry && contact_[i] && !releasing_[i] &&
                                 direction_[i] * contact_output_[i] < 0;
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
