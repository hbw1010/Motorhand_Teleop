#pragma once

#include "types.hpp"

#include <filesystem>
#include <memory>

namespace motorhand::native {

// Stateful, single-threaded official Hand2 optimizer. Angles are radians.
class WujiRetarget {
public:
    WujiRetarget(const std::filesystem::path& yaml_path, bool left);
    ~WujiRetarget();
    WujiRetarget(WujiRetarget&&) noexcept;
    WujiRetarget& operator=(WujiRetarget&&) noexcept;
    WujiRetarget(const WujiRetarget&) = delete;
    WujiRetarget& operator=(const WujiRetarget&) = delete;

    Vec20 solve(const Points& canonical_points_m);
    Points hand_points(const Vec20& actual_joint_rad);
    void reset();

private:
    friend class ForceJacobianColumns;
    void configure_differentiation();
    void set_initial_hint(const Vec20* hint);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace motorhand::native
