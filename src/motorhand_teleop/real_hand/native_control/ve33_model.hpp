#pragma once

#include "types.hpp"

#include <array>
#include <filesystem>
#include <memory>

namespace motorhand::native {

// CAD joint and motor coordinates are radians; points are canonical metres.
// Evaluations are pure; pass the previous valid pose to follow its branch.
class Ve33Model {
public:
    explicit Ve33Model(const std::filesystem::path& asset_root, bool left);
    ~Ve33Model();
    Ve33Model(Ve33Model&&) noexcept;
    Ve33Model& operator=(Ve33Model&&) noexcept;
    Ve33Model(const Ve33Model&) = delete;
    Ve33Model& operator=(const Ve33Model&) = delete;

    GlovePose zero();
    GlovePose evaluate_motors(const Vec12& motors, const GlovePose* seed = nullptr);
    GlovePose evaluate_joints(const Vec12& joints, const GlovePose* seed = nullptr);
    GlovePose evaluate_tree(const Vec68& tree);
    std::array<std::array<double, 2>, 12> active_joint_bounds(const GlovePose& pose);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace motorhand::native
