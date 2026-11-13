#pragma once

#include "types.hpp"
#include "wuji_identity.hpp"

#include <cstdint>
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

namespace motorhand::native {

class Trajectory;

struct SessionOptions {
    std::filesystem::path asset_root;
    std::filesystem::path retarget_yaml;
    std::filesystem::path glove_mapping;
    std::filesystem::path linear_feedback;
    std::filesystem::path contact_feedback;
    std::string wuji_serial;
    std::string approved_firmware;
    bool glove_output = true;
    bool force_feedback = true;
    bool linear_force_feedback = false;
    bool driver_impedance_feedback = false; // force-contact CLI默认启用；其他调用保留旧模式。
    std::filesystem::path driver_impedance_config;
    double feedback_hardness = 0.3; // 人工设定，非物体硬度估计。
    bool left = false;
    // 独立力反馈遥操：两侧各自回零，使用固定的手套零位重定向参考。
    // 推理/回放接管不启用，不改变其保持抓握的过渡语义。
    bool zero_aligned_start = false;
    // 可选取消标志由调用方持有，须活过Session；CLI使用无锁原子信号标志。
    const std::atomic<bool>* stop_requested = nullptr;
    std::uint64_t transition_ns = 500'000'000;
    // Inference: the glove shadow follows the Wuji *command* (true) or the measured
    // Wuji position (false).  Following the command keeps a grasp's squeeze (command
    // deeper than the blocked fingers) in the glove pose, so the forward solve at
    // takeover reproduces the command instead of releasing the object.
    bool inverse_follows_command = true;
    // Takeover handover: the remaining difference between the last command and the
    // forward solve is kept as a per-joint offset, so the switch itself does not move
    // the hand.  An offset decays only as the operator moves that joint (by
    // exp(-|motion| / takeover_offset_motion_rad)); takeover_offset_motion_rad = 0
    // restores the legacy 0.5 s blend.  takeover_offset_tau_ns > 0 adds a time decay
    // (the hand then also re-aligns while the operator holds still); 0 = off (default,
    // to be judged on hardware).
    double takeover_offset_motion_rad = 0.10;
    std::uint64_t takeover_offset_tau_ns = 0;
};

struct SessionTick {
    HandFrame hand{};
    GloveFrame glove{};
    Vec20 wuji_target{};
    Vec12 glove_target{};  // MIT model position in inference; nominal torque in takeover.
    bool takeover{};
    bool inverse_fresh{};
    bool force_geometry_valid{};
    std::uint64_t target_deadline_ns{};
    GlovePose glove_pose{};
    Vec20 virtual_hand_nm{};  // Software virtual torque, not a measurement.
    std::array<bool, 20> contact{};
    ContactPhase contact_phase = ContactPhase::not_configured;
    bool force_commanded{};
    std::uint64_t jacobian_source_ns{};
    std::uint64_t glove_output_source_ns{}; // Original input time for the completed output.
    std::uint64_t cycle_started_ns{};
    std::uint64_t wuji_submit_ns{};  // Host SDK call returned, not wire receipt.
    std::uint64_t glove_submit_ns{}; // Last axis submitted to its serial device.
    std::uint64_t cycle_completed_ns{};
    bool linear_force{};
    Vec20 wuji_load_fraction{};  // Current-normalized estimate for linear mode.
    HandFrame feedback_input_hand{};
    Vec20 feedback_input_target{};
    std::uint64_t feedback_request_id{};
    Vec12 glove_feedforward_nm{}; // 实际MIT编码的tau_ff；阻抗模式不等于总PD。
    Vec12 glove_projected_nm{}, glove_requested_nm{};
    Vec20 filtered_current_a{};
    double contact_reflection_gain{};
    std::array<double, 240> hand_to_motor{};
    Vec20 wuji_projection_current_a{};
    std::uint64_t wuji_projection_received_ns{};
    bool driver_impedance{};
    Vec12 glove_friction_nm{}, glove_driver_pd_nm{};
    Vec12 glove_driver_target_rad{}, glove_driver_kp{}, glove_driver_kd{}; // 模型轴，发布时转raw。
    double feedback_hardness{};
    std::array<std::uint64_t,12> glove_driver_estimate_received_ns{},glove_driver_velocity_received_ns{};
    DriverImpedanceObservation glove_impedance_observation{};
    std::uint64_t contact_evaluated_ns{}, impedance_planned_ns{};
};

// The session owns both devices. Construction/connection are read-only;
// only enable() takes control and homes the glove. All time fields are
// original host monotonic timestamps, never refreshed after computation.
class Session {
public:
    explicit Session(SessionOptions options);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    const WujiIdentity& connect();
    // Read a fresh device sample without enabling or changing its timestamp.
    HandFrame read_wuji_state();
    // Validate every recorded frame and its device identity before enable.
    void validate_trajectory(const Trajectory& trajectory) const;
    void enable();
    // The caller creates an inference sample timestamp when the model or
    // trajectory actually supplied the target. Playback uses the current
    // trajectory interpolation time, not the last device frame's timestamp.
    SessionTick step_inference(const Vec20& target, std::uint64_t source_ns);
    SessionTick step_takeover();
    void request_takeover(bool takeover);
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace motorhand::native
