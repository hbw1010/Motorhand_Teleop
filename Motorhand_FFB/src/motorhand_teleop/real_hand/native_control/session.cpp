#include "session.hpp"

#include "encos_driver.hpp"
#include "force_controller.hpp"
#include "driver_impedance.hpp"
#include "glove_output.hpp"
#include "inverse_worker.hpp"
#include "jacobian_worker.hpp"
#include "trajectory.hpp"
#include "ve33_model.hpp"
#include "wuji_device.hpp"
#include "wuji_retarget.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace motorhand::native {
namespace {
constexpr std::uint64_t timeout_ns = 100'000'000;
constexpr std::uint64_t inverse_margin_ns = 30'000'000;
using Clock = std::chrono::steady_clock;
std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}
std::uint64_t deadline(std::uint64_t source) {
    if (!source || source > std::numeric_limits<std::uint64_t>::max() - timeout_ns)
        throw std::runtime_error("invalid source timestamp");
    return source + timeout_ns;
}
void require_fresh(std::uint64_t stamp, std::uint64_t now) {
    if (!stamp || stamp > now || now - stamp >= timeout_ns) throw std::runtime_error("stale or future source timestamp");
}
void require_deadline(std::uint64_t until) {
    if (now_ns() >= until) throw std::runtime_error("original control command deadline expired");
}
template<class A> bool finite(const A& a) {
    return std::all_of(a.begin(), a.end(), [](double x) { return std::isfinite(x); });
}
nlohmann::json json_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open configuration: " + path.string());
    nlohmann::json data; stream >> data; return data;
}
double smooth(std::uint64_t elapsed, std::uint64_t duration) {
    const double t = duration == 0 ? 1.0 : std::min(1.0, static_cast<double>(elapsed) / static_cast<double>(duration));
    return t*t*t*(10.0 + t*(-15.0 + 6.0*t));
}
template<std::size_t N>
std::array<double, N> blend(const std::array<double, N>& a, const std::array<double, N>& b, double x) {
    std::array<double, N> out{};
    for (std::size_t i = 0; i < N; ++i) out[i] = a[i] + x * (b[i] - a[i]);
    if (!finite(out)) throw std::runtime_error("non-finite transition target");
    return out;
}
std::string describe(std::exception_ptr e) {
    if (!e) return {};
    try { std::rethrow_exception(e); } catch (const std::exception& ex) { return ex.what(); } catch (...) { return "unknown exception"; }
}
}

struct Session::Impl {
    SessionOptions options;
    WujiIdentity identity;
    std::unique_ptr<WujiDevice> hand;
    std::unique_ptr<EncosDriver> glove;
    std::unique_ptr<Ve33Model> model;
    std::unique_ptr<WujiRetarget> retarget;
    std::unique_ptr<InverseWorker> inverse;
    std::unique_ptr<JacobianWorker> jacobian;
    std::unique_ptr<ForceController> force;
    std::unique_ptr<DriverImpedancePlanner> impedance;
    std::unique_ptr<GloveOutputWorker> output_worker;
    std::optional<GlovePose> last_pose;
    std::optional<InverseWorkerResult> inverse_result;
    std::optional<JacobianSnapshot> jacobian_result;
    Vec20 lower{}, upper{}, last_hand_target{}, hand_anchor{};
    Vec20 takeover_offset{}, takeover_last_raw{};
    Vec20 zero_reference{};
    bool takeover_has_raw = false;
    std::uint64_t takeover_offset_clock = 0;
    Vec12 glove_anchor{};
    bool connected = false, enabled = false, closed = false, fault = false;
    bool requested_takeover = false, mode_takeover = false, mode_initialized = false;
    bool has_hand_target = false, inverse_was_fresh = false, need_glove_anchor = true;
    bool glove_output_started = false, hand_output_started = false;
    mutable bool trajectory_rejected = false;
    std::uint64_t transition_elapsed = 0, transition_clock = 0, glove_transition_start = 0;
    std::uint64_t inverse_submit_stamp = 0, inverse_last_id = 0, inverse_min_id = 0, inverse_consumed_id = 0;
    std::uint64_t inverse_wait_start = 0, jac_submit_stamp = 0;

    explicit Impl(SessionOptions o) : options(std::move(o)) {
        if (options.force_feedback && !options.glove_output)
            throw std::invalid_argument("force feedback requires glove output");
        if (options.zero_aligned_start && !options.glove_output)
            throw std::invalid_argument("zero-aligned start requires glove output for homing");
        if (options.driver_impedance_feedback && (!options.force_feedback || options.linear_force_feedback))
            throw std::invalid_argument("driver impedance requires contact feedback");
        if (!std::isfinite(options.feedback_hardness) || options.feedback_hardness<0 || options.feedback_hardness>1)
            throw std::invalid_argument("feedback hardness must be in [0,1]");
        if (!options.transition_ns) throw std::invalid_argument("transition duration must be positive");
    }
    std::string cleanup() noexcept {
        std::string errors;
        auto attempt = [&](const char* name, auto&& action) {
            try { action(); } catch (...) {
                if (!errors.empty()) errors += "; ";
                errors += std::string(name) + ": " + describe(std::current_exception());
            }
        };
        if (output_worker) {
            attempt("Encos output worker close", [&] { output_worker->close(); });
            output_worker.reset();
        }
        if (glove && glove_output_started) attempt("Encos disable", [&] { glove->disable(); });
        glove_output_started = false;
        if (hand && hand_output_started) attempt("Wuji disable", [&] { hand->disable(); });
        hand_output_started = false;
        enabled = false;
        if (hand) attempt("Wuji close", [&] { hand->close(); });
        if (glove) attempt("Encos close", [&] { glove->close(); });
        jacobian.reset();
        inverse.reset();
        connected = false; closed = true;
        return errors;
    }
    [[noreturn]] void fail(std::exception_ptr original) {
        fault = true;
        const auto cause = describe(original);
        const auto cleanup_error = cleanup();
        throw std::runtime_error(cause + (cleanup_error.empty() ? "" : "; shutdown: " + cleanup_error));
    }
    HandFrame read_hand(bool require_enabled) {
        const RealHandState raw = hand->read();
        const auto now = now_ns();
        require_fresh(raw.timestamp_ns, now);
        if ((require_enabled && raw.phase != 2) || (!require_enabled && raw.phase != 1 && raw.phase != 2))
            throw std::runtime_error("Wuji not in expected connected/enabled state");
        HandFrame result;
        result.received_ns = raw.timestamp_ns;
        result.has_velocity = raw.has_velocity != 0; result.has_effort = raw.has_effort != 0;
        for (int i = 0; i < 20; ++i) {
            if (!raw.online[i] || (require_enabled && !raw.enabled[i]) ||
                !std::isfinite(raw.positions[i]) || !std::isfinite(raw.lower_limits[i]) ||
                !std::isfinite(raw.upper_limits[i]) || raw.lower_limits[i] >= raw.upper_limits[i])
                throw std::runtime_error("Wuji feedback incomplete, faulted, or limits invalid");
            lower[i] = raw.lower_limits[i]; upper[i] = raw.upper_limits[i];
            result.position_rad[i] = raw.positions[i];
            result.velocity_rad_s[i] = raw.velocities_rad_s[i]; result.effort_a[i] = raw.effort_a[i];
        }
        if ((result.has_velocity && !finite(result.velocity_rad_s)) || (result.has_effort && !finite(result.effort_a)))
            throw std::runtime_error("non-finite Wuji feedback");
        if (options.force_feedback && !result.has_effort) throw std::runtime_error("Wuji effort feedback unavailable");
        return result;
    }
    GloveFrame read_glove() {
        auto frame = glove->read();
        const auto now = now_ns();
        require_fresh(frame.oldest_ns, now);
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (auto stamp : frame.received_ns) { require_fresh(stamp, now); oldest = std::min(oldest, stamp); }
        if (oldest != frame.oldest_ns || !finite(frame.motor_rad) || !finite(frame.raw_degrees) || !finite(frame.measured_current_a))
            throw std::runtime_error("invalid Encos feedback");
        return frame;
    }
    GlovePose pose_for(const GloveFrame& frame) {
        auto pose = model->evaluate_motors(frame.motor_rad, last_pose ? &*last_pose : nullptr);
        last_pose = pose;
        return pose;
    }
    void check_target(const Vec20& target) const {
        if (!finite(target)) throw std::invalid_argument("non-finite Wuji target");
        for (int i = 0; i < 20; ++i)
            if (target[i] < lower[i] || target[i] > upper[i])
                throw std::invalid_argument("Wuji target exceeds joint limits");
    }
    void enter_mode(bool takeover, const HandFrame& hand_frame, const GloveFrame& glove_frame) {
        const bool was_takeover = mode_initialized && mode_takeover;
        mode_takeover = takeover; mode_initialized = true;
        hand_anchor = has_hand_target ? last_hand_target : hand_frame.position_rad;
        transition_elapsed = 0; transition_clock = now_ns();
        inverse_result.reset(); inverse_min_id = inverse_last_id; inverse_consumed_id = inverse_min_id;
        inverse_submit_stamp = 0; inverse_was_fresh = false; need_glove_anchor = true; inverse_wait_start = 0;
        glove_anchor = glove_frame.motor_rad;
        if (takeover && options.force_feedback) {
            output_worker->enter_force();
            glove_output_started = true;
        } else if (!takeover && was_takeover && options.force_feedback) {
            const auto until = std::min(deadline(hand_frame.received_ns), deadline(glove_frame.oldest_ns));
            require_deadline(until);
            output_worker->enter_position(until);
        } else if (takeover && options.glove_output && !options.force_feedback) {
            output_worker->close();
            output_worker.reset();
            glove->disable();
        } else if (!takeover && options.glove_output && !output_worker) {
            output_worker = std::make_unique<GloveOutputWorker>(*glove, nullptr);
        }
    }
    double hand_blend(std::uint64_t now, bool advance) {
        if (now < transition_clock) throw std::runtime_error("transition clock moved backwards");
        if (advance) {
            const auto dt = now - transition_clock;
            transition_elapsed += std::min(dt, options.transition_ns - std::min(transition_elapsed, options.transition_ns));
        }
        transition_clock = now;
        return smooth(transition_elapsed, options.transition_ns);
    }
};

Session::Session(SessionOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Session::~Session() { if (impl_ && !impl_->closed) impl_->cleanup(); }
const WujiIdentity& Session::connect() {
    auto& s = *impl_;
    if (s.closed || s.fault || s.connected) throw std::runtime_error("session cannot connect in current state");
    try {
        s.identity = inspect_wuji(s.options.wuji_serial, s.options.left);
        if ((!s.options.wuji_serial.empty() && s.identity.serial != s.options.wuji_serial) || s.identity.left != s.options.left ||
            s.identity.firmware != s.options.approved_firmware)
            throw std::runtime_error("Wuji preflight identity or firmware mismatch");
        const auto mapping_config = json_file(s.options.glove_mapping);
        if (!mapping_config.contains("calibrated") || mapping_config.at("calibrated") != true)
            throw std::runtime_error("Encos mapping is not calibrated");
        double jac_step = 0.01;
        if (s.options.force_feedback) {
            const auto cfg = json_file(s.options.linear_feedback);
            if (cfg.at("wuji_serial_number").get<std::string>() != s.identity.serial ||
                cfg.at("glove_hand").get<std::string>() != (s.options.left ? "left" : "right"))
                throw std::runtime_error("force feedback serial or hand mismatch");
            jac_step = cfg.value("jacobian_step_rad", 0.01);
        }
        s.glove = std::make_unique<EncosDriver>(s.options.glove_mapping.string(),
            s.options.force_feedback ? s.options.linear_feedback.string() : std::string{});
        const auto cal = s.glove->calibration();
        Vec12 signs{}, scales{}, lo{}, hi{};
        std::array<int, 12> raw_to_model{};
        std::array<bool, 12> seen{};
        std::array<std::array<double, 2>, 12> bounds{};
        for (int i = 0; i < 12; ++i) {
            const auto& m = cal[i]; const int raw = m.raw_motor_id - 1;
            if (raw < 0 || raw >= 12 || seen[raw] || !std::isfinite(m.sign) || !std::isfinite(m.scale) ||
                m.sign == 0 || m.scale == 0 || !std::isfinite(m.model_min_rad) || !std::isfinite(m.model_max_rad) ||
                m.model_min_rad >= m.model_max_rad || (s.options.force_feedback && !m.has_torque_limit))
                throw std::runtime_error("invalid Encos motor calibration");
            seen[raw] = true; signs[raw] = m.sign; scales[raw] = m.scale; raw_to_model[raw] = i + 1;
            lo[i] = m.model_min_rad; hi[i] = m.model_max_rad; bounds[i] = {lo[i], hi[i]};
        }
        s.model = std::make_unique<Ve33Model>(s.options.asset_root, s.options.left);
        s.retarget = std::make_unique<WujiRetarget>(s.options.retarget_yaml, s.options.left);
        if (s.options.zero_aligned_start) {
            // 在无输出时收敛模型零位的优化器先验；不能把任意启动姿态作为零位。
            const auto zero = s.model->zero();
            for (int i = 0; i < 30; ++i) s.zero_reference = s.retarget->solve(zero.points_m);
        }
        if (s.options.force_feedback) {
            s.force = std::make_unique<ForceController>(s.options.contact_feedback.string(), s.options.linear_feedback.string(), signs, scales, raw_to_model, s.options.linear_force_feedback);
            s.force->set_motor_limits(lo, hi);
            if (s.options.driver_impedance_feedback)
                s.impedance = std::make_unique<DriverImpedancePlanner>(
                    load_driver_impedance_config(s.options.driver_impedance_config.string()),s.options.feedback_hardness,cal);
        }
        s.hand = std::make_unique<WujiDevice>(s.identity, s.options.approved_firmware);
        s.hand->connect();
        s.glove->connect();
        s.read_hand(false);
        auto glove = s.read_glove();
        s.pose_for(glove);
        if (s.options.glove_output)
            s.inverse = std::make_unique<InverseWorker>(s.options.asset_root, s.options.retarget_yaml, s.options.left, bounds);
        if (s.options.force_feedback)
            s.jacobian = std::make_unique<JacobianWorker>(s.options.asset_root, s.options.retarget_yaml, s.options.left, jac_step);
        s.connected = true;
        return s.identity;
    } catch (...) { s.fail(std::current_exception()); }
}
HandFrame Session::read_wuji_state() {
    auto& s = *impl_;
    try {
        if (!s.connected || s.closed || s.fault) throw std::runtime_error("session is not connected");
        auto frame = s.read_hand(s.enabled);
        if (!frame.has_velocity || !frame.has_effort)
            throw std::runtime_error("Wuji velocity or current feedback unavailable");
        return frame;
    } catch (...) { s.fail(std::current_exception()); }
}
void Session::validate_trajectory(const Trajectory& trajectory) const {
    auto& s = *impl_;
    if (!s.connected || s.enabled || s.closed || s.fault) throw std::runtime_error("trajectory validation requires connected, disabled session");
    s.trajectory_rejected = true;
    if (trajectory.hand() != (s.identity.left ? "left" : "right") || trajectory.serial() != s.identity.serial ||
        trajectory.firmware() != s.identity.firmware) throw std::runtime_error("trajectory identity or firmware mismatch");
    trajectory.validate_limits(s.lower, s.upper);
    s.trajectory_rejected = false;
}
void Session::enable() {
    auto& s = *impl_;
    if (!s.connected || s.enabled || s.closed || s.fault || s.trajectory_rejected)
        throw std::runtime_error("session cannot enable in current state");
    try {
        if (s.options.stop_requested && s.options.stop_requested->load())
            throw std::runtime_error("启动已取消");
        if (s.options.glove_output) { s.glove_output_started = true; s.glove->home(); }
        if (s.options.stop_requested && s.options.stop_requested->load())
            throw std::runtime_error("启动已取消");
        if (s.options.glove_output)
            s.output_worker = std::make_unique<GloveOutputWorker>(*s.glove, s.force.get(), s.impedance.get());
        s.hand_output_started = true; s.hand->enable();
        if (s.options.zero_aligned_start) s.hand->home(s.options.stop_requested);
        const auto h = s.read_hand(true); const auto g = s.read_glove();
        s.pose_for(g);
        s.last_hand_target = h.position_rad; s.has_hand_target = false;
        s.mode_initialized = false; s.enabled = true;
    } catch (...) { s.fail(std::current_exception()); }
}
void Session::request_takeover(bool takeover) {
    auto& s = *impl_;
    if (!s.enabled || s.closed || s.fault) throw std::runtime_error("session is not enabled");
    s.requested_takeover = takeover;
}

SessionTick Session::step_inference(const Vec20& input, std::uint64_t source_ns) {
    auto& s = *impl_;
    try {
        if (!s.enabled || s.closed || s.fault || s.requested_takeover) throw std::runtime_error("session is not in inference mode");
        const auto cycle_start = now_ns();
        require_fresh(source_ns, now_ns());
        Vec20 candidate = input; s.check_target(candidate);
        auto h = s.read_hand(true); auto g = s.read_glove(); auto pose = s.pose_for(g);
        if (!s.mode_initialized || s.mode_takeover) {
            s.enter_mode(false, h, g);
            // Mode transition may block while Encos releases force: re-read all sources.
            h = s.read_hand(true); g = s.read_glove(); pose = s.pose_for(g);
        }
        bool fresh_inverse = !s.options.glove_output;
        if (s.inverse) {
            while (auto done = s.inverse->poll()) {
                if (done->request_id <= s.inverse_min_id || done->request_id <= s.inverse_consumed_id) continue;
                const auto now = now_ns();
                if (done->source_ns > h.received_ns || done->source_ns > now) throw std::runtime_error("inverse source timestamp invalid");
                s.inverse_consumed_id = done->request_id;
                if (now - done->source_ns >= timeout_ns) continue;
                if (!done->result.success) throw std::runtime_error("glove inverse failed: " + done->result.failure_reason);
                s.inverse_result = std::move(*done);
            }
            if (h.received_ns > s.inverse_submit_stamp) {
                HandFrame source = h;
                if (s.options.inverse_follows_command && s.has_hand_target) source.position_rad = s.last_hand_target;
                s.inverse_last_id = s.inverse->submit(source, pose, h.received_ns);
                s.inverse_submit_stamp = h.received_ns;
            }
            auto now = now_ns();
            fresh_inverse = s.inverse_result && s.inverse_result->source_ns <= now &&
                now - s.inverse_result->source_ns < timeout_ns - inverse_margin_ns;
            if (!fresh_inverse) {
                s.need_glove_anchor = true;
                if (!s.inverse_wait_start) s.inverse_wait_start = now;
                else if (now - s.inverse_wait_start >= 2'000'000'000ULL) throw std::runtime_error("waiting for fresh glove inverse exceeded two seconds");
            } else {
                s.inverse_wait_start = 0;
                if (s.need_glove_anchor) {
                    g = s.read_glove(); pose = s.pose_for(g);
                    s.glove_anchor = g.motor_rad; s.glove_transition_start = now_ns(); s.need_glove_anchor = false;
                }
            }
        }
        auto now = now_ns();
        const double alpha = s.hand_blend(now, fresh_inverse && s.inverse_was_fresh);
        s.inverse_was_fresh = fresh_inverse;
        Vec20 target = fresh_inverse ? blend(s.hand_anchor, candidate, alpha) :
            (s.has_hand_target ? s.last_hand_target : h.position_rad);
        s.check_target(target);
        auto until = std::min({deadline(source_ns), deadline(h.received_ns), deadline(g.oldest_ns)});
        std::uint64_t glove_until = until;
        Vec12 requested_glove{};
        if (s.inverse && fresh_inverse) {
            glove_until = std::min(glove_until, deadline(s.inverse_result->source_ns));
            const auto elapsed = now_ns() - s.glove_transition_start;
            requested_glove = blend(s.glove_anchor, s.inverse_result->result.pose.motor_rad, smooth(elapsed, s.options.transition_ns));
        }
        const auto completed = s.output_worker ? s.output_worker->poll() : std::nullopt;
        if (completed && completed->mode != GloveOutputMode::position)
            throw std::runtime_error("old glove output crossed mode barrier");
        require_deadline(until);
        if (s.inverse && fresh_inverse) require_deadline(glove_until);
        s.hand->send(target, until);
        const auto wuji_submit = now_ns();
        s.last_hand_target = target; s.has_hand_target = true;
        if (s.inverse && fresh_inverse) {
            require_deadline(glove_until);
            s.output_worker->submit_position(requested_glove, glove_until - timeout_ns, glove_until);
            s.glove_output_started = true;
        }
        const Vec12 glove_target = completed ? completed->applied : Vec12{};
        const std::uint64_t glove_submit = completed ? completed->last_write_ns : 0;
        SessionTick tick{h, g, target, glove_target, false, fresh_inverse, false, until};
        tick.glove_pose = pose;
        tick.force_commanded = false;
        tick.glove_output_source_ns = completed ? completed->source_ns : 0;
        tick.cycle_started_ns = cycle_start; tick.wuji_submit_ns = wuji_submit;
        tick.glove_submit_ns = glove_submit; tick.cycle_completed_ns = now_ns();
        return tick;
    } catch (...) { s.fail(std::current_exception()); }
}

SessionTick Session::step_takeover() {
    auto& s = *impl_;
    try {
        if (!s.enabled || s.closed || s.fault || !s.requested_takeover) throw std::runtime_error("session is not in takeover mode");
        const auto cycle_start = now_ns();
        auto h = s.read_hand(true); auto g = s.read_glove(); auto pose = s.pose_for(g);
        if (!s.mode_initialized || !s.mode_takeover) {
            s.enter_mode(true, h, g);
            h = s.read_hand(true); g = s.read_glove(); pose = s.pose_for(g);
            s.transition_clock = now_ns();
            // The forward solver was idle during inference: start it (warm start and
            // norm_delta prior) at the command being handed over, not a stale pose.
            s.retarget->seed(s.options.zero_aligned_start ? s.zero_reference : s.hand_anchor);
            s.takeover_has_raw = false;
        }
        if (s.jacobian) {
            if (auto completed = s.jacobian->poll()) s.jacobian_result = std::move(*completed);
            if (g.oldest_ns > s.jac_submit_stamp) {
                s.jacobian->submit(pose, g.oldest_ns); s.jac_submit_stamp = g.oldest_ns;
            }
        }
        Vec20 target = s.retarget->solve(pose.points_m);
        if (!s.options.zero_aligned_start) s.check_target(target);
        std::array<bool, 20> saturated{};
        double zero_blend = 1.0;
        if (s.options.zero_aligned_start) {
            // 固定参考只减去零手套的重定向姿态；不随累计运动衰减，回零可重复。
            for (int i = 0; i < 20; ++i) {
                target[i] -= s.zero_reference[i];
                saturated[i] = target[i] <= s.lower[i] || target[i] >= s.upper[i];
                target[i] = std::clamp(target[i], s.lower[i], s.upper[i]);
            }
            zero_blend = s.hand_blend(now_ns(), true);
            target = blend(s.hand_anchor, target, zero_blend);
        } else if (s.options.takeover_offset_motion_rad > 0) {
            const auto now = now_ns();
            if (!s.takeover_has_raw) {
                for (int i = 0; i < 20; ++i) s.takeover_offset[i] = s.hand_anchor[i] - target[i];
                s.takeover_has_raw = true;
            } else {
                const double dt = now > s.takeover_offset_clock ? double(now - s.takeover_offset_clock) : 0.0;
                const double time_decay = s.options.takeover_offset_tau_ns ?
                    std::exp(-dt / double(s.options.takeover_offset_tau_ns)) : 1.0;
                for (int i = 0; i < 20; ++i) {
                    const double motion = std::abs(target[i] - s.takeover_last_raw[i]);
                    s.takeover_offset[i] *= time_decay * std::exp(-motion / s.options.takeover_offset_motion_rad);
                }
            }
            s.takeover_offset_clock = now;
            s.takeover_last_raw = target;
            for (int i = 0; i < 20; ++i)
                target[i] = std::clamp(target[i] + s.takeover_offset[i], s.lower[i], s.upper[i]);
        } else {
            target = blend(s.hand_anchor, target, s.hand_blend(now_ns(), true));
        }
        s.check_target(target);
        const auto until = std::min(deadline(h.received_ns), deadline(g.oldest_ns));
        const auto completed = s.output_worker ? s.output_worker->poll() : std::nullopt;
        if (completed && completed->mode != GloveOutputMode::force)
            throw std::runtime_error("old glove output crossed mode barrier");
        require_deadline(until);
        s.hand->send(target, until);
        const auto wuji_submit = now_ns();
        s.last_hand_target = target; s.has_hand_target = true;
        if (s.force) {
            require_deadline(until);
            auto effective_jacobian = s.jacobian_result;
            if (s.options.zero_aligned_start && effective_jacobian) {
                for (int row = 0; row < 20; ++row)
                    for (int col = 0; col < 12; ++col)
                        effective_jacobian->jacobian[row * 12 + col] *= saturated[row] ? 0.0 : zero_blend;
            }
            s.output_worker->submit_force(target, h, g, pose, effective_jacobian, until);
        }
        const bool computed = completed && completed->force_computed;
        const ForceResult feedback = computed ? completed->force : ForceResult{};
        const Vec12 actual = completed ? completed->applied : Vec12{};
        const std::uint64_t glove_submit = completed ? completed->last_write_ns : 0;
        SessionTick tick{h, g, target, actual, true, true, computed && feedback.active, until};
        tick.glove_pose = pose; tick.virtual_hand_nm = feedback.virtual_hand_nm; tick.contact = feedback.contact;
        tick.linear_force = s.options.linear_force_feedback;
        tick.wuji_load_fraction = feedback.wuji_load_fraction;
        tick.glove_projected_nm = feedback.projected_motor_nm;
        tick.glove_requested_nm = feedback.model_motor_nm;
        tick.driver_impedance = s.options.driver_impedance_feedback;
        if (completed) {
            tick.feedback_input_hand = completed->feedback_input_hand;
            tick.feedback_input_target = completed->feedback_input_target;
            tick.feedback_request_id = completed->request_id;
            tick.contact_evaluated_ns = completed->contact_evaluated_ns;
            tick.impedance_planned_ns = completed->impedance_planned_ns;
            tick.glove_feedforward_nm = completed->driver_impedance ?
                completed->impedance.command.feedforward_nm : completed->applied;
        }
        tick.feedback_hardness = s.options.feedback_hardness;
        if (completed && completed->driver_impedance) {
            tick.glove_impedance_observation = completed->impedance_observation;
            tick.glove_friction_nm = completed->impedance.friction_nm;
            tick.glove_driver_pd_nm = completed->impedance.pd_nm;
            tick.glove_driver_target_rad = completed->impedance.command.target_rad;
            tick.glove_driver_kp = completed->impedance.command.kp;
            tick.glove_driver_kd = completed->impedance.command.kd;
            tick.glove_driver_estimate_received_ns = completed->impedance.estimate_received_ns;
            tick.glove_driver_velocity_received_ns = completed->impedance.command.velocity_received_ns;
        }
        tick.filtered_current_a = feedback.filtered_current_a;
        tick.contact_reflection_gain = feedback.reflection_gain;
        tick.hand_to_motor = feedback.hand_to_motor;
        tick.wuji_projection_current_a = feedback.projection_current_a;
        tick.wuji_projection_received_ns = feedback.projection_received_ns;
        tick.contact_phase = completed ? (computed ? feedback.phase : ContactPhase::waiting_geometry) :
                                         ContactPhase::not_configured;
        tick.force_commanded = completed.has_value();
        tick.glove_output_source_ns = completed ? completed->source_ns : 0;
        tick.jacobian_source_ns = completed ? completed->jacobian_source_ns : 0;
        tick.cycle_started_ns = cycle_start; tick.wuji_submit_ns = wuji_submit;
        tick.glove_submit_ns = glove_submit; tick.cycle_completed_ns = now_ns();
        return tick;
    } catch (...) { s.fail(std::current_exception()); }
}
void Session::stop() {
    auto& s = *impl_;
    if (s.closed) return;
    const auto errors = s.cleanup();
    if (!errors.empty()) { s.fault = true; throw std::runtime_error("session shutdown: " + errors); }
}
} // namespace motorhand::native
