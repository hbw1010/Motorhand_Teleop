#pragma once
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <string>
namespace motorhand::native {
struct ControlOptions {
    std::filesystem::path asset_root;
    std::filesystem::path glove_mapping;
    std::filesystem::path linear_feedback;
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
    // 可选取消标志由调用方持有，须活过会话；CLI使用无锁原子信号标志。
    const std::atomic<bool>* stop_requested = nullptr;
    std::uint64_t transition_ns = 500'000'000;
    // Inference: the glove shadow follows the hand *command* (true) or the measured
    // hand position (false).  Following the command keeps a grasp's squeeze (command
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

inline void validate_control_options(const ControlOptions& options) {
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
}
