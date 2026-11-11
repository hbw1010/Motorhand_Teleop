// The glove's output: what Motorhand's own session does with the glove (session.cpp: homing, step_inference's
// glove following, step_takeover's force feedback), from Tianji's Wuji hand as the node passes it on the worker's
// stdin instead of from Motorhand's own Wuji device. Its own parts unchanged: EncosDriver, InverseWorker,
// JacobianWorker, ForceController, GloveOutputWorker.
//
//   force   the hand follows this glove (its final target is one this glove sent): force feedback, Motorhand's
//           contact impedance: where the Wuji hand is blocked (loaded, still, short of its target) the glove's
//           motors push back. Entered only once the hand has reached this glove's targets (every joint within
//           0.06 rad, the contact threshold, for 0.2 s): Motorhand's session blends the hand there itself before
//           its force acts, here the hand driver does (its homing, its 3 s handover), and a hand still on its way
//           would look blocked
//   follow  the hand follows something else (a model: its newest target, under 0.1 s old, is not one this glove
//           sent): the glove's motors move it to the hand's pose (the inverse retarget), from where it was over
//           0.5 s, each pushing no harder than its force-feedback cap (the hold is kp x the motor's error: its
//           target stays within cap / kp of where the motor is)
//   idle    no fresh hand state, or no fresh target at all (nothing authorized drives the hand: before r, after a
//           stop): nothing driven, the motors released (the driver's command deadline passes, ~0.1 s)
#pragma once

#include "encos_driver.hpp"
#include "force_controller.hpp"
#include "glove_output.hpp"
#include "inverse_worker.hpp"
#include "jacobian_worker.hpp"
#include "types.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace motorhand_glove {
using namespace motorhand::native;

// What the node tells the worker, a JSON object a line: the hand's state as received at monotonic ns,
// {"hand": {"ns", "q", "dq", "effort"}}, and its final target, {"target": {"ns", "q", "ours"}} (ours: one this glove
// sent); the 20 joints in the command order, the WujiRetarget's. Anything else is ignored.
class HandFeed {
public:
    struct Snapshot {
        std::optional<HandFrame> hand;
        std::optional<Vec20> target;
        std::uint64_t target_ns = 0;
        bool ours = false;
    };
    void take(const std::string& line);
    Snapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    Snapshot latest_;
};

struct ControlOptions {
    std::filesystem::path root;  // the Motorhand_FFB checkout
    std::filesystem::path retarget_yaml;  // the WujiRetarget's (the inverse's, the Jacobian's); empty: Motorhand's own
    bool left = false;
    bool follow = false;           // follow mode
    double follow_max_nm = 2.0;    //   its push on each motor without force limits (with them: their caps)
    double follow_kp = 45.0;       //   the driver's position stiffness, Nm/rad (EncosDriver::set_position_kp)
    std::filesystem::path limits;  // force mode: the limits config (torque caps and slew, the Wuji hand it is for)
    std::filesystem::path contact; //   and the contact-impedance config
    std::string serial;            // the Wuji hand's serial, checked against the limits config's; empty: not checked
};

// Whether the next GloveControl skips homing (EncosDriver::mark_homed instead of home): --no-home, or this worker has
// homed the glove once already (a reconnect after a motor fault resumes from where the hand is).
void set_skip_homing(bool skip);
bool skip_homing();

class GloveControl {
public:
    // The glove connected; homing starts (in the background: the glove is read meanwhile).
    GloveControl(EncosDriver& glove, const ControlOptions& options);
    ~GloveControl();  // stops homing and output; the driver's disable and close stay the caller's
    GloveControl(const GloveControl&) = delete;
    GloveControl& operator=(const GloveControl&) = delete;

    // One frame: the glove's reading and its pose (model evaluated), the latest of the feed. Throws when the
    // glove's output cannot go on (failed homing, a latched output): reconnect.
    void step(const GloveFrame& glove, const GlovePose& pose, const HandFeed::Snapshot& feed);
    std::vector<std::string> events();  // since the last call: homing, mode changes, recoveries
    nlohmann::json stats();             // since the last call

private:
    enum class Mode { idle, follow, force };
    void enter(Mode mode, const GloveFrame& glove);
    void push(const GloveFrame& glove, const GlovePose& pose, const HandFrame& hand, const Vec20& target);
    void follow(const GloveFrame& glove, const GlovePose& pose, const HandFrame& hand);
    void record(const GloveOutputResult& done);
    void recover(const std::string& why);

    EncosDriver& glove_;
    ControlOptions options_;
    std::unique_ptr<ForceController> force_;
    std::unique_ptr<JacobianWorker> jacobian_;
    std::unique_ptr<InverseWorker> inverse_;
    std::unique_ptr<GloveOutputWorker> output_;
    std::future<void> homing_;
    std::vector<std::string> events_;
    std::deque<std::uint64_t> failures_;

    Mode mode_ = Mode::idle, wanted_ = Mode::idle;
    std::uint64_t wanted_since_ = 0;
    std::uint64_t caught_up_since_ = 0;  // force waits for the hand to have reached this glove's targets
    bool told_waiting_ = false;
    // force
    std::optional<JacobianSnapshot> jacobian_result_;
    std::uint64_t jacobian_submitted_ = 0, jacobian_clock_ = 0, jacobian_every_ns_ = 0, force_source_ = 0;
    // follow
    std::optional<InverseWorkerResult> inverse_result_;
    std::uint64_t inverse_last_id_ = 0, inverse_min_id_ = 0, inverse_submitted_ = 0, anchor_ns_ = 0;
    bool need_anchor_ = true;
    Vec12 anchor_{};
    Vec12 follow_reach_{};  // how far (model rad) a motor's target may be from where it is: cap / kp, per motor
    // this stats window
    struct Window {
        long frames = 0, homing = 0, force = 0, contact = 0, geometry = 0, follow = 0, follow_waiting = 0;
        long inverse_failed = 0;
        double peak_nm = 0, max_current_a = 0, max_error_rad = 0;
        int peak_axis = -1;
    } window_;
};
}  // namespace motorhand_glove
