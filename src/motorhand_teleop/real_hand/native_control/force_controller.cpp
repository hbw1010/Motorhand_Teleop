#include "force_controller.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace motorhand::native {
namespace {
using Json = nlohmann::json;
constexpr double kHalfLsb = 10.0 / 4095.0;
constexpr double kTau = 6.283185307179586476925286766559;

Json load(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("cannot open force feedback configuration: " + path);
    Json result;
    file >> result;
    if (!result.is_object()) throw std::invalid_argument("feedback configuration must be an object");
    return result;
}
double number(const Json& value) {
    if (!value.is_number()) throw std::invalid_argument("feedback value must be a number");
    const double x = value.get<double>();
    if (!std::isfinite(x)) throw std::invalid_argument("feedback value must be finite");
    return x;
}
double positive(const Json& j, const char* key, double fallback) {
    const double x = j.contains(key) ? number(j.at(key)) : fallback;
    if (!(x > 0)) throw std::invalid_argument(std::string(key) + " must be positive");
    return x;
}
template<std::size_t N>
std::array<double, N> array(const Json& j, const char* key) {
    const auto& value = j.at(key);
    if (!value.is_array() || value.size() != N) throw std::invalid_argument(std::string(key) + " has invalid size");
    std::array<double, N> out{};
    for (std::size_t i = 0; i < N; ++i) out[i] = number(value[i]);
    return out;
}
template<class T> bool finite(const T& values) {
    return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
bool fresh(std::uint64_t stamp, std::uint64_t now, double max_age) {
    return stamp != 0 && stamp <= now && static_cast<double>(now - stamp) * 1e-9 <= max_age;
}
void identity(const Json& j) {
    if (!j.at("schema_version").is_number_integer() || j.at("schema_version") != 1 ||
        !j.at("estimate_acknowledged").is_boolean() || j.at("estimate_acknowledged") != true ||
        !j.at("source_id").is_string() || j.at("source_id").get<std::string>().find_first_not_of(" \t\n\r") == std::string::npos)
        throw std::invalid_argument("feedback estimate must be explicitly acknowledged");
}
} // namespace

ForceController::ForceController(const std::string& contact_path, const std::string& linear_path,
                                 const Vec12& sign, const Vec12& scale,
                                 const std::array<int, 12>& raw_to_model, bool linear_mode)
    : linear_mode_(linear_mode) {
    const auto c = load(contact_path), l = load(linear_path);
    identity(c); identity(l);
    if (l.at("mode") != "max_torque_linear_estimate" ||
        (l.at("glove_hand") != "left" && l.at("glove_hand") != "right") ||
        !l.at("wuji_serial_number").is_string() ||
        l.at("wuji_serial_number").get<std::string>().find_first_not_of(" \t\n\r") == std::string::npos)
        throw std::invalid_argument("invalid linear feedback identity");
    current_on_ = array<20>(c, "current_on_a"); current_off_ = array<20>(c, "current_off_a");
    error_on_ = array<20>(c, "error_on_rad"); error_off_ = array<20>(c, "error_off_rad");
    stalled_ = array<20>(c, "stalled_speed_rad_s"); stiffness_ = array<20>(c, "stiffness_nm_per_rad");
    damping_ = array<20>(c, "damping_nm_s_per_rad"); bias_ = array<20>(l, "wuji_current_bias_a");
    const auto max_current = array<20>(l, "wuji_max_current_a");
    const auto effort_sign = array<20>(l, "wuji_effort_sign");
    for (int i = 0; i < 20; ++i) {
        if (!(current_on_[i] > current_off_[i] && current_off_[i] >= 0 && error_on_[i] > error_off_[i] && error_off_[i] >= 0 && stalled_[i] > 0 && stiffness_[i] > 0 && damping_[i] >= 0 && max_current[i] > 0 && (effort_sign[i] == 1 || effort_sign[i] == -1)))
            throw std::invalid_argument("invalid contact or current thresholds");
    }
    dwell_ = positive(c, "dwell_s", .15); release_dwell_ = positive(c, "release_dwell_s", .06);
    current_hz_ = positive(c, "current_filter_hz", 5); output_hz_ = positive(c, "output_filter_hz", 8);
    max_gap_ = positive(c, "max_sample_gap_s", .1); load_scale_ = positive(c, "load_weight_scale_a", .5);
    opening_deadband_ = positive(c, "opening_speed_deadband_rad_s", .02);
    velocity_limit_ = positive(c, "relative_velocity_limit_rad_s", 2);
    reflection_gain_ = positive(c, "reflection_gain", 1);
    if (opening_deadband_ >= velocity_limit_) throw std::invalid_argument("invalid opening deadband");
    gain_ = positive(l, "gain", 0); ramp_ = positive(l, "ramp_duration_s", .5);
    feedback_age_ = std::min(.1, positive(l, "feedback_max_age_s", .1));
    jac_age_ = positive(l, "jacobian_max_age_s", .5);
    jac_pose_delta_ = positive(l, "jacobian_max_pose_delta_rad", .25);
    positive(l, "jacobian_refresh_interval_s", .1); positive(l, "jacobian_step_rad", .01);
    if (gain_ > 1) throw std::invalid_argument("feedback gain exceeds one");
    linear_parameters_ = {max_current, bias_, effort_sign, gain_};
    const auto cap = array<12>(l, "glove_max_output_torque_nm");
    const auto slew = array<12>(l, "max_torque_slew_nm_s");
    std::array<bool, 12> seen{};
    for (int raw = 0; raw < 12; ++raw) {
        const int model = raw_to_model[raw] - 1;
        if (model < 0 || model >= 12 || seen[model] || !std::isfinite(sign[raw]) ||
            sign[raw] == 0 || !std::isfinite(scale[raw]) || scale[raw] == 0 ||
            !std::isfinite(sign[raw] * scale[raw]) ||
            cap[raw] <= kHalfLsb || cap[raw] > 10 || slew[raw] <= 0)
            throw std::invalid_argument("invalid motor binding or torque limits");
        seen[model] = true;
        max_torque_[model] = std::nextafter(cap[raw] - kHalfLsb, 0.0) / std::abs(sign[raw] * scale[raw]);
        max_ack_torque_[model] = cap[raw] / std::abs(sign[raw] * scale[raw]);
        codec_zero_[model] = max_ack_torque_[model] - max_torque_[model];
        max_slew_[model] = slew[raw] / std::abs(sign[raw] * scale[raw]);
        if (!std::isfinite(max_torque_[model]) || !std::isfinite(max_slew_[model]) || !std::isfinite(max_ack_torque_[model]))
            throw std::invalid_argument("motor scale produces invalid limits");
    }
}
void ForceController::set_motor_limits(const Vec12& lower, const Vec12& upper) {
    if (!finite(lower) || !finite(upper)) throw std::invalid_argument("invalid motor limits");
    for (int i = 0; i < 12; ++i) if (lower[i] >= upper[i]) throw std::invalid_argument("invalid motor limits");
    lower_ = lower; upper_ = upper; limits_set_ = true;
}
void ForceController::clear_axis(int i) {
    active_[i] = candidate_set_[i] = release_set_[i] = false;
    direction_[i] = candidate_direction_[i] = 0; output_[i] = 0;
}
void ForceController::clear_contact() {
    have_source_ = have_hand_ = have_current_ = false;
    for (int i = 0; i < 20; ++i) clear_axis(i);
}
void ForceController::reset(std::uint64_t now_ns) {
    clear_contact(); history_.fill(0); reset_ns_ = history_ns_ = now_ns; reset_set_ = true; pending_ = false;
    linear_geometry_active_ = false;
}

ForceResult ForceController::step(const Vec20& target, const HandFrame& hand, const GloveFrame& glove,
                                  const GlovePose& pose, const std::array<double, 240>& jac,
                                  const Vec12& jac_q, std::uint64_t jac_ns, std::uint64_t now) {
    ForceResult result;
    auto stop = [&]() -> ForceResult {
        clear_contact(); pending_ = false;
        throw std::runtime_error("invalid or stale force feedback; stop output");
    };
    if (!reset_set_ || !limits_set_ || now < reset_ns_ || now < history_ns_ || !hand.has_effort ||
        !fresh(hand.received_ns, now, feedback_age_) || !fresh(glove.oldest_ns, now, feedback_age_) ||
        !finite(target) || !finite(hand.position_rad) || !finite(hand.effort_a) ||
        (hand.has_velocity && !finite(hand.velocity_rad_s)) || !finite(glove.motor_rad) ||
        !finite(pose.motor_rad) || !finite(pose.joint_rad) || !finite(pose.motor_jacobian) ||
        !finite(jac) || !finite(jac_q)) return stop();
    bool geometry_valid = fresh(jac_ns, now, jac_age_) && jac_ns <= glove.oldest_ns;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (int i = 0; i < 12; ++i) {
        if (!fresh(glove.received_ns[i], now, feedback_age_) ||
            std::abs(pose.motor_rad[i] - glove.motor_rad[i]) > 1e-7) return stop();
        if (std::abs(pose.joint_rad[i] - jac_q[i]) > jac_pose_delta_) geometry_valid = false;
        oldest = std::min(oldest, glove.received_ns[i]);
    }
    if (oldest != glove.oldest_ns) return stop();
    const std::uint64_t source = std::min(hand.received_ns, glove.oldest_ns);
    if ((have_source_ && source < source_ns_) || (have_hand_ && hand.received_ns < hand_ns_)) return stop();
    using Mat12 = Eigen::Matrix<double, 12, 12, Eigen::RowMajor>;
    using Mat20x12 = Eigen::Matrix<double, 20, 12, Eigen::RowMajor>;
    const Eigen::Map<const Mat12> motor(pose.motor_jacobian.data());
    const Eigen::Map<const Mat20x12> mapping(jac.data());
    Eigen::FullPivLU<Eigen::Matrix<double, 12, 12>> lu(motor.transpose());
    if (lu.rank() != 12) return stop();
    if (!linear_mode_) {
    if (have_source_ && static_cast<double>(source - source_ns_) * 1e-9 > max_gap_) clear_contact();
    const bool new_frame = !have_source_ || source > source_ns_;
    const double dt = have_source_ ? static_cast<double>(source - source_ns_) * 1e-9 : 0;
    Vec20 measured = hand.velocity_rad_s, target_velocity{};
    bool have_velocity = hand.has_velocity, have_target_velocity = false;
    if (new_frame) {
        if (dt > 0 && have_source_) {
            for (int i = 0; i < 20; ++i) target_velocity[i] = (target[i] - last_target_[i]) / dt;
            have_target_velocity = true;
        }
        if (!have_velocity && have_hand_ && hand.received_ns > hand_ns_) {
            const double hand_dt = static_cast<double>(hand.received_ns - hand_ns_) * 1e-9;
            for (int i = 0; i < 20; ++i) measured[i] = (hand.position_rad[i] - last_position_[i]) / hand_dt;
            have_velocity = true;
        }
        const double alpha = -std::expm1(-kTau * current_hz_ * dt);
        for (int i = 0; i < 20; ++i) {
            const double load = std::abs(hand.effort_a[i] - bias_[i]);
            if (!have_current_) filtered_current_[i] = load;
            else filtered_current_[i] += alpha * (load - filtered_current_[i]);
        }
        have_current_ = true;
    }
    if (!finite(filtered_current_) || (have_velocity && !finite(measured)) || (have_target_velocity && !finite(target_velocity))) return stop();
    const double alpha = -std::expm1(-kTau * output_hz_ * dt);
    for (int i = 0; i < 20; ++i) {
        bool observable = false;
        for (int j = 0; j < 12; ++j) observable |= jac[12*i+j] != 0;
        if (!observable) { clear_axis(i); continue; }
        const double error = target[i] - hand.position_rad[i];
        if (!std::isfinite(error)) return stop();
        if (active_[i]) {
            if (direction_[i] * error <= error_off_[i] || filtered_current_[i] <= current_off_[i]) {
                output_[i] = 0;
                if (!release_set_[i]) { release_set_[i] = true; release_ns_[i] = source; }
                if (static_cast<double>(source - release_ns_[i]) * 1e-9 >= release_dwell_) clear_axis(i);
                continue;
            }
            release_set_[i] = false;
        } else {
            const int dir = (error > 0) - (error < 0);
            if (!new_frame) {
                if (std::abs(error) < error_on_[i] || dir != candidate_direction_[i] ||
                    (have_velocity && std::abs(measured[i]) > stalled_[i])) clear_axis(i);
                continue;
            }
            if (std::abs(error) < error_on_[i] || filtered_current_[i] < current_on_[i] ||
                !have_velocity || std::abs(measured[i]) > stalled_[i]) { clear_axis(i); continue; }
            if (!candidate_set_[i] || dir != candidate_direction_[i]) {
                candidate_set_[i] = true; candidate_ns_[i] = source; candidate_direction_[i] = dir;
            }
            if (static_cast<double>(source - candidate_ns_[i]) * 1e-9 < dwell_) continue;
            active_[i] = true; direction_[i] = dir; candidate_set_[i] = false; candidate_direction_[i] = 0;
        }
        if (!new_frame) {
            if (have_source_ && direction_[i] * (target[i] - last_target_[i]) < 0) output_[i] = 0;
            continue;
        }
        const int n = direction_[i];
        const double penetration = std::max(n * error - error_off_[i], 0.0);
        double relative = 0;
        if (have_target_velocity && have_velocity)
            relative = std::clamp(n * (target_velocity[i] - measured[i]), -velocity_limit_, velocity_limit_);
        if (relative < -opening_deadband_) { output_[i] = 0; continue; }
        const double load = std::min(filtered_current_[i] / load_scale_, 20.0);
        const double speed = std::min(std::abs(relative) / stalled_[i], 20.0);
        const double load_weight = -std::expm1(-load * load);
        const double damping_weight = .6 / (1 + std::exp(-load)) + .4 / (1 + std::exp(-speed));
        const double requested = -n * std::max(load_weight * stiffness_[i] * penetration + damping_weight * damping_[i] * relative, 0.0);
        output_[i] += alpha * (requested - output_[i]);
    }
    if (!finite(output_)) return stop();
    if (new_frame) {
        have_source_ = true; source_ns_ = source; last_target_ = target;
        if (!have_hand_ || hand.received_ns > hand_ns_) {
            have_hand_ = true; hand_ns_ = hand.received_ns; last_position_ = hand.position_rad;
        }
    }
    } else {
        have_source_ = true;
        source_ns_ = source;
        for (int i = 0; i < 20; ++i)
            result.wuji_load_fraction[i] = std::clamp(
                (hand.effort_a[i] - linear_parameters_.wuji_current_bias_a[i]) *
                    linear_parameters_.wuji_effort_sign[i] / linear_parameters_.wuji_max_current_a[i],
                -1.0, 1.0);
    }
    Eigen::Matrix<double,12,1> projected = Eigen::Matrix<double,12,1>::Zero();
    if (geometry_valid) {
        if (linear_mode_) {
            const auto linear = project_max_torque_linear(
                jac, pose.motor_jacobian, hand.effort_a, linear_parameters_, max_torque_);
            for (int i = 0; i < 12; ++i) projected[i] = linear.model_torque_nm[i];
        } else {
            Eigen::Map<const Eigen::Matrix<double,20,1>> virtual_torque(output_.data());
            const Eigen::Matrix<double,12,1> joint_torque = mapping.transpose() * virtual_torque;
            projected = (gain_ * reflection_gain_) * lu.solve(joint_torque);
        }
        if (!projected.allFinite()) return stop();
    }
    if (linear_mode_) {
        if (geometry_valid && !linear_geometry_active_) reset_ns_ = now;
        linear_geometry_active_ = geometry_valid;
    }
    const double x = std::min(1.0, static_cast<double>(now - reset_ns_) * 1e-9 / ramp_);
    const double ramp = x*x*x*(10 + x*(-15 + 6*x));
    const double history_dt = static_cast<double>(now - history_ns_) * 1e-9;
    for (int i = 0; i < 12; ++i) {
        double wanted = std::clamp(projected[i] * ramp, -max_torque_[i], max_torque_[i]);
        if (linear_mode_) {
            if (geometry_valid) wanted = std::clamp(wanted, history_[i] - max_slew_[i] * history_dt,
                                                     history_[i] + max_slew_[i] * history_dt);
        } else if (history_[i] * wanted < 0) wanted = 0;
        else wanted = std::copysign(std::min(std::abs(wanted), std::abs(history_[i]) + max_slew_[i] * history_dt), wanted);
        if ((pose.motor_rad[i] <= lower_[i] && wanted < 0) || (pose.motor_rad[i] >= upper_[i] && wanted > 0)) wanted = 0;
        result.model_motor_nm[i] = wanted;
    }
    result.virtual_hand_nm = output_; result.contact = active_; result.active = geometry_valid; result.linear = linear_mode_;
    if (linear_mode_) {
        result.phase = ContactPhase::not_configured;
    } else if (!geometry_valid) {
        result.phase = ContactPhase::waiting_geometry;
    } else {
        bool contacting = false, releasing = false, arming = false;
        for (int i = 0; i < 20; ++i) {
            contacting |= active_[i] && !release_set_[i];
            releasing |= active_[i];
            arming |= candidate_set_[i];
        }
        result.phase = contacting ? ContactPhase::contact : releasing ? ContactPhase::releasing :
            arming ? ContactPhase::arming : ContactPhase::free;
    }
    pending_output_ = result.model_motor_nm; pending_ns_ = now; pending_ = true;
    return result;
}
void ForceController::acknowledge_torque(const Vec12& actual) {
    if (!pending_ || !finite(actual)) throw std::runtime_error("invalid torque acknowledgement");
    for (int i = 0; i < 12; ++i) {
        if (std::abs(actual[i]) > max_ack_torque_[i] + 1e-12)
            throw std::runtime_error("acknowledged torque exceeds configured limit");
    }
    for (int i = 0; i < 12; ++i) {
        const bool zero_bin = std::abs(std::abs(actual[i]) - codec_zero_[i]) <= 1e-12 + 1e-9 * codec_zero_[i];
        const bool same_small = pending_output_[i] * actual[i] >= 0 &&
                                std::abs(pending_output_[i]) <= codec_zero_[i] + 1e-12;
        const double value = zero_bin ? (same_small ? pending_output_[i] :
            (pending_output_[i] * actual[i] < 0 ? 0.0 : actual[i])) : actual[i];
        history_[i] = std::clamp(value, -max_torque_[i], max_torque_[i]);
    }
    history_ns_ = pending_ns_; pending_ = false;
}
} // namespace motorhand::native
