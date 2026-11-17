#pragma once

#include "types.hpp"
#include "common/hand.hpp"

#include <filesystem>
#include <memory>

namespace motorhand::native {

// Stateful, single-threaded official Hand2 optimizer. Angles are radians.
class WujiRetarget : public HandModel<20> {
public:
    WujiRetarget(const std::filesystem::path& yaml_path, bool left);
    ~WujiRetarget();
    WujiRetarget(WujiRetarget&&) noexcept;
    WujiRetarget& operator=(WujiRetarget&&) noexcept;
    WujiRetarget(const WujiRetarget&) = delete;
    WujiRetarget& operator=(const WujiRetarget&) = delete;

    Vec20 solve(const Points& canonical_points_m);
    Points hand_points(const Vec20& actual_joint_rad);
    // Per task point (MediaPipe 21 order), the factor the forward solve applies to the
    // glove's wrist-relative point before matching Hand2 (segment_scaling; 1 for points
    // the forward objective does not position).  hand_points / scale is therefore the
    // glove geometry the forward solve maps onto a given Hand2 pose.
    std::array<double, 21> task_scales() const;
    // Warm start and temporal prior (norm_delta) at a known Hand2 pose, e.g. the
    // command being sent when the forward solve takes over from another source.
    void seed(const Vec20& joint_rad);
    void reset();
    std::unique_ptr<HandModel<20>> create_solver(bool differentiation=false) const override;
    void set_initial_hint(const Vec20* hint) override;

private:

    void configure_differentiation();

    std::filesystem::path yaml_path_;
    bool left_{};
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace motorhand::native
