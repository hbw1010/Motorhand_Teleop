#include "glove_control.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <atomic>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace motorhand_glove {
namespace {
std::atomic<bool> skip_homing_{false};
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kLife = 100'000'000;          // a source's life: Motorhand's session's command timeout
constexpr std::uint64_t kInverseMargin = 30'000'000;  // an inverse result is used until 70 ms old (its session's)
constexpr std::uint64_t kTransition = 500'000'000;    // following starts from where the glove is, over 0.5 s
constexpr std::uint64_t kSettle = 30'000'000;         // a new mode must be wanted this long (idle: at once)
constexpr double kCaughtUpRad = 0.06;                 // force starts with the hand this close to every target
constexpr std::uint64_t kCaughtUpFor = 200'000'000;   //   for this long
constexpr std::array<const char*, 12> kAxes = {
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip", "index_mcp_flex", "index_pip",
    "middle_mcp_flex", "middle_pip", "ring_mcp_flex", "ring_pip", "pinky_mcp_flex", "pinky_pip"};

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}
bool fresh(std::uint64_t stamp, std::uint64_t now) { return stamp && stamp <= now && now - stamp < kLife; }
double smooth(std::uint64_t elapsed, std::uint64_t duration) {
    const double t = std::min(1.0, static_cast<double>(elapsed) / static_cast<double>(duration));
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}
nlohmann::json json_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    nlohmann::json data;
    stream >> data;
    return data;
}
Vec20 joints(const nlohmann::json& values) {
    if (!values.is_array() || values.size() != 20) throw std::invalid_argument("not 20 joints");
    Vec20 out{};
    for (int i = 0; i < 20; ++i) {
        out[i] = values[i].get<double>();
        if (!std::isfinite(out[i])) throw std::invalid_argument("non-finite");
    }
    return out;
}
double percent(long part, long whole) { return whole ? std::round(1000.0 * part / whole) / 10.0 : 0.0; }
double hundredths(double value) { return std::round(value * 100.0) / 100.0; }
const char* name(int mode) { return mode == 2 ? "force" : mode == 1 ? "follow" : "idle"; }
}  // namespace

void HandFeed::take(const std::string& line) {
    try {
        const auto message = nlohmann::json::parse(line);
        if (!message.is_object()) return;
        if (message.contains("hand")) {
            const auto& h = message.at("hand");
            HandFrame frame;
            frame.received_ns = h.at("ns").get<std::uint64_t>();
            frame.position_rad = joints(h.at("q"));
            frame.velocity_rad_s = joints(h.at("dq"));
            frame.effort_a = joints(h.at("effort"));
            frame.has_velocity = frame.has_effort = true;
            std::lock_guard<std::mutex> lock(mutex_);
            if (!latest_.hand || frame.received_ns > latest_.hand->received_ns) latest_.hand = frame;
        } else if (message.contains("target")) {
            const auto& t = message.at("target");
            const Vec20 q = joints(t.at("q"));
            const auto ns = t.at("ns").get<std::uint64_t>();
            const bool ours = t.at("ours").get<bool>();
            std::lock_guard<std::mutex> lock(mutex_);
            if (ns > latest_.target_ns) {
                latest_.target = q;
                latest_.target_ns = ns;
                latest_.ours = ours;
            }
        }
    } catch (const std::exception&) {
        // not a complete hand state or target (a joint state still NaN: not JSON): skipped
    }
}

HandFeed::Snapshot HandFeed::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

void set_skip_homing(bool skip) { skip_homing_.store(skip); }
bool skip_homing() { return skip_homing_.load(); }

GloveControl::GloveControl(EncosDriver& glove, const ControlOptions& options) : glove_(glove), options_(options) {
    const std::string side = options_.left ? "left" : "right";
    const auto cal = glove_.calibration();
    Vec12 signs{}, scales{}, lower{}, upper{};
    std::array<int, 12> raw_to_model{};
    std::array<std::array<double, 2>, 12> bounds{};
    for (int i = 0; i < 12; ++i) {  // as Motorhand's session binds them
        const auto& m = cal[i];
        const int raw = m.raw_motor_id - 1;
        if (raw < 0 || raw >= 12 || (!options_.limits.empty() && !m.has_torque_limit))
            throw std::runtime_error("invalid glove motor calibration");
        signs[raw] = m.sign;
        scales[raw] = m.scale;
        raw_to_model[raw] = i + 1;
        lower[i] = m.model_min_rad;
        upper[i] = m.model_max_rad;
        bounds[i] = {lower[i], upper[i]};
        // Follow's hold is the driver's MIT position control with no torque term: kp x (target - where the motor
        // is), at the motor, whose angle is the model axis' / (sign x scale). Its targets stay within the reach
        // where that is at most this motor's force-feedback cap (the driver's, from the limits), or follow_max_nm.
        const double cap = m.has_torque_limit ? m.raw_torque_limit_nm : options_.follow_max_nm;
        follow_reach_[i] = cap * std::abs(m.sign * m.scale) / options_.follow_kp;
        if (!std::isfinite(follow_reach_[i]) || !(follow_reach_[i] > 0))
            throw std::runtime_error("invalid follow torque cap for glove motor " + std::to_string(m.raw_motor_id));
    }
    const auto assets = options_.root / "assets" / "hand_motor_ve33";
    const auto yaml = options_.retarget_yaml.empty()
                          ? options_.root / "config" / "retargeting" / "official" / (side + ".yaml")
                          : options_.retarget_yaml;
    if (!options_.limits.empty()) {
        const auto limits = json_file(options_.limits);
        if (limits.at("glove_hand").get<std::string>() != side)
            throw std::runtime_error("the force feedback limits " + options_.limits.string() + " are not the " + side +
                                     " hand's");
        const auto serial = limits.at("wuji_serial_number").get<std::string>();
        if (!options_.serial.empty() && serial != options_.serial)
            throw std::runtime_error("the force feedback limits are Wuji hand " + serial + "'s, this one is " +
                                     options_.serial);
        jacobian_every_ns_ = static_cast<std::uint64_t>(limits.value("jacobian_refresh_interval_s", 0.1) * 1e9);
        force_ = std::make_unique<ForceController>(options_.contact.string(), options_.limits.string(), signs, scales,
                                                   raw_to_model, false);
        force_->set_motor_limits(lower, upper);
        jacobian_ = std::make_unique<JacobianWorker>(assets, yaml, options_.left,
                                                     limits.value("jacobian_step_rad", 0.01));
    }
    if (options_.follow) inverse_ = std::make_unique<InverseWorker>(assets, yaml, options_.left, bounds);
    const bool skip = skip_homing();
    if (skip) {
        events_.push_back("glove not homed (homed earlier by this worker, or --no-home): its output starts where it is");
    } else {
        events_.push_back("homing the glove: its motors to the zero pose at 5 rpm (each within 0.3 rad); keep the "
                          "hand relaxed");
    }
    // Motorhand's own tolerance is 0.15 rad: worn, the thumb's abduction rests ~9 deg off its zero, never arrives,
    // and its motors stall against the hand until they fault.
    homing_ = std::async(std::launch::async, [this, skip] {
        if (skip) glove_.mark_homed();
        else glove_.home(0.3);
    });
}

GloveControl::~GloveControl() {
    if (homing_.valid()) {
        if (homing_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            try { glove_.disable(); } catch (...) {}  // ends the homing
        }
        try { homing_.get(); } catch (...) {}
    }
    if (output_) {
        try { output_->close(); } catch (...) {}
    }
}

void GloveControl::step(const GloveFrame& glove, const GlovePose& pose, const HandFeed::Snapshot& feed) {
    ++window_.frames;
    if (!output_) {
        if (homing_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++window_.homing;
            return;
        }
        const bool skipped = skip_homing();
        homing_.get();  // a failed homing throws: reconnected and homed again
        set_skip_homing(true);  // once: a reconnect resumes from where the hand is
        output_ = std::make_unique<GloveOutputWorker>(glove_, force_.get());
        std::string what = skipped ? "glove output ready:" : "glove homed:";
        if (force_) what += " force feedback while the hand follows this glove;";
        if (inverse_) what += " the glove follows the hand while it follows something else;";
        what.pop_back();
        events_.push_back(what);
    }
    try {
        if (auto done = output_->poll()) record(*done);
        const auto now = now_ns();
        const bool hand = feed.hand && fresh(feed.hand->received_ns, now);
        const bool targeted = feed.target && fresh(feed.target_ns, now);
        const bool ours = targeted && feed.ours;
        if (hand) {
            for (int i = 0; i < 20; ++i) {
                window_.max_current_a = std::max(window_.max_current_a, std::abs(feed.hand->effort_a[i]));
                if (ours)
                    window_.max_error_rad = std::max(window_.max_error_rad,
                                                     std::abs((*feed.target)[i] - feed.hand->position_rad[i]));
            }
        }
        Mode want = Mode::idle;
        if (hand && ours && force_) want = Mode::force;
        // Something else drives the hand (a model): no target at all is nothing authorized (before r, after a stop)
        else if (hand && targeted && !ours && inverse_) want = Mode::follow;
        if (want == Mode::force && mode_ != Mode::force) {
            double behind = 0;
            for (int i = 0; i < 20; ++i)
                behind = std::max(behind, std::abs((*feed.target)[i] - feed.hand->position_rad[i]));
            if (behind >= kCaughtUpRad) caught_up_since_ = 0;
            else if (!caught_up_since_) caught_up_since_ = now;
            if (!caught_up_since_ || now - caught_up_since_ < kCaughtUpFor) {
                if (!told_waiting_)
                    events_.push_back("glove: force feedback once the hand has reached this glove's targets (now " +
                                      std::to_string(static_cast<int>(std::round(behind * 57.29577951308232))) +
                                      " deg behind)");
                told_waiting_ = true;
                want = mode_ == Mode::follow ? Mode::follow : Mode::idle;  // as it was meanwhile
            }
        } else {
            caught_up_since_ = 0;
            told_waiting_ = false;
        }
        if (want != wanted_) {
            wanted_ = want;
            wanted_since_ = now;
        }
        if (want != mode_ && (want == Mode::idle || now - wanted_since_ >= kSettle)) enter(want, glove);
        if (mode_ == Mode::force) push(glove, pose, *feed.hand, *feed.target);
        else if (mode_ == Mode::follow) follow(glove, pose, *feed.hand);
    } catch (const std::exception& e) {
        recover(e.what());
    }
}

void GloveControl::enter(Mode mode, const GloveFrame& glove) {
    if (mode_ == Mode::force) output_->enter_position(glove.oldest_ns + kLife);  // a zero torque frame first
    if (mode == Mode::force) {
        output_->enter_force();  // the controller reset: no reaction yet, ramped in
        force_source_ = 0;
    }
    if (mode == Mode::follow) {
        inverse_result_.reset();
        inverse_min_id_ = inverse_last_id_;
        inverse_submitted_ = 0;
        need_anchor_ = true;
    }
    mode_ = mode;
    events_.push_back(mode == Mode::force    ? "glove: force feedback (the hand follows this glove)"
                      : mode == Mode::follow ? "glove: following the hand (it follows something else)"
                                             : "glove: idle (no hand state, or nothing drives the hand), its motors "
                                               "released");
}

void GloveControl::push(const GloveFrame& glove, const GlovePose& pose, const HandFrame& hand, const Vec20& target) {
    if (auto done = jacobian_->poll()) jacobian_result_ = std::move(*done);
    auto now = now_ns();
    if (glove.oldest_ns > jacobian_submitted_ && now - jacobian_clock_ >= jacobian_every_ns_) {
        jacobian_->submit(pose, glove.oldest_ns);
        jacobian_submitted_ = glove.oldest_ns;
        jacobian_clock_ = now;
    }
    // New samples only: on a repeated one the controller zeroes an axis whose target moved back at all.
    const auto source = std::min(hand.received_ns, glove.oldest_ns);
    if (source <= force_source_) return;
    const auto until = source + kLife;
    if (now_ns() >= until) return;
    output_->submit_force(target, hand, glove, pose, jacobian_result_, until);
    force_source_ = source;
}

void GloveControl::follow(const GloveFrame& glove, const GlovePose& pose, const HandFrame& hand) {
    ++window_.follow;
    while (auto done = inverse_->poll()) {
        if (done->request_id <= inverse_min_id_) continue;  // asked before this mode began
        if (!done->result.success) {
            ++window_.inverse_failed;
            continue;
        }
        inverse_result_ = std::move(*done);
    }
    if (hand.received_ns > inverse_submitted_) {
        inverse_last_id_ = inverse_->submit(hand, pose, hand.received_ns);
        inverse_submitted_ = hand.received_ns;
    }
    const auto now = now_ns();
    if (!inverse_result_ || inverse_result_->source_ns > now ||
        now - inverse_result_->source_ns >= kLife - kInverseMargin) {
        need_anchor_ = true;
        ++window_.follow_waiting;
        return;
    }
    if (need_anchor_) {
        anchor_ = glove.motor_rad;
        anchor_ns_ = now;
        need_anchor_ = false;
    }
    const double x = smooth(now - anchor_ns_, kTransition);
    const auto& goal = inverse_result_->result.pose.motor_rad;
    Vec12 position{};
    for (int i = 0; i < 12; ++i) {
        position[i] = anchor_[i] + x * (goal[i] - anchor_[i]);
        position[i] = std::clamp(position[i], glove.motor_rad[i] - follow_reach_[i],
                                 glove.motor_rad[i] + follow_reach_[i]);  // the hold within its cap
    }
    const auto until = std::min({hand.received_ns, glove.oldest_ns, inverse_result_->source_ns}) + kLife;
    if (now >= until) return;
    output_->submit_position(position, until - kLife, until);
}

void GloveControl::record(const GloveOutputResult& done) {
    if (done.mode != GloveOutputMode::force) return;
    ++window_.force;
    if (done.force.phase == ContactPhase::contact) ++window_.contact;
    if (!done.force_computed || done.force.phase == ContactPhase::waiting_geometry) ++window_.geometry;
    for (int i = 0; i < 12; ++i) {
        if (std::abs(done.applied[i]) > window_.peak_nm) {
            window_.peak_nm = std::abs(done.applied[i]);
            window_.peak_axis = i;
        }
    }
}

void GloveControl::recover(const std::string& why) {
    const auto now = now_ns();
    failures_.push_back(now);
    while (!failures_.empty() && now - failures_.front() > 10'000'000'000ULL) failures_.pop_front();
    if (failures_.size() >= 5) throw std::runtime_error("the glove's output failed 5 times in 10 s, last: " + why);
    try { output_->close(); } catch (...) {}  // its own failure: it has disabled the glove already
    output_.reset();
    glove_.disable();  // throws if the output latched: reconnected then
    output_ = std::make_unique<GloveOutputWorker>(glove_, force_.get());
    mode_ = wanted_ = Mode::idle;
    jacobian_result_.reset();
    events_.push_back("glove output stopped (" + why + "): its motors released; resuming");
}

std::vector<std::string> GloveControl::events() { return std::exchange(events_, {}); }

nlohmann::json GloveControl::stats() {
    const auto& w = window_;
    nlohmann::json out = {{"mode", output_ ? name(static_cast<int>(mode_)) : "homing"},
                          {"max_current_a", hundredths(w.max_current_a)},
                          {"max_error_rad", hundredths(w.max_error_rad)}};
    if (force_) {
        out["contact_pct"] = percent(w.contact, w.force);
        out["geometry_wait_pct"] = percent(w.geometry, w.force);
        out["peak_nm"] = hundredths(w.peak_nm);
        out["peak_axis"] = w.peak_axis >= 0 ? kAxes[w.peak_axis] : "";
    }
    if (inverse_) {
        out["follow_waiting_pct"] = percent(w.follow_waiting, w.follow);
        out["inverse_failed"] = w.inverse_failed;
    }
    window_ = Window{};
    return out;
}
}  // namespace motorhand_glove
