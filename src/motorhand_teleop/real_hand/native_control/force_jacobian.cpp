#include "force_jacobian.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace motorhand::native {
namespace {
void validate_request(const Vec12& q, const std::vector<int>& columns, double step_rad) {
    for (double value : q)
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite glove joint");
    if (!std::isfinite(step_rad) || step_rad <= 0)
        throw std::invalid_argument("Jacobian step must be positive and finite");
    std::array<bool, 12> seen{};
    for (int column : columns) {
        if (column < 0 || column >= 12 || seen[column])
            throw std::invalid_argument("Jacobian columns must be distinct indices in [0, 12)");
        seen[column] = true;
    }
}
}  // namespace

ForceJacobianColumns::ForceJacobianColumns(const std::filesystem::path& asset_root,
                                           const std::filesystem::path& yaml_path,
                                           bool left)
    : model_(asset_root, left), retarget_(yaml_path, left) {
    retarget_.configure_differentiation();
}

std::vector<double> ForceJacobianColumns::compute(const Vec12& q,
                                                   const std::vector<int>& columns,
                                                   double step_rad) {
    validate_request(q, columns, step_rad);
    if (columns.empty()) return {};
    const GlovePose centre = model_.evaluate_joints(q, seed_ ? &*seed_ : nullptr);
    return compute_centre(q, centre, columns, step_rad);
}

std::vector<double> ForceJacobianColumns::compute(const GlovePose& base_pose,
                                                   const std::vector<int>& columns,
                                                   double step_rad) {
    validate_request(base_pose.joint_rad, columns, step_rad);
    if (columns.empty()) return {};
    // Reconstruct all derived data from the actual tree rather than trusting
    // points or joint coordinates carried alongside it.
    const GlovePose centre = model_.evaluate_tree(base_pose.tree_rad);
    for (std::size_t i = 0; i < 12; ++i) {
        if (std::abs(centre.joint_rad[i] - base_pose.joint_rad[i]) > 1e-8)
            throw std::invalid_argument("glove pose joints do not match CAD tree");
    }
    return compute_centre(base_pose.joint_rad, centre, columns, step_rad);
}

std::vector<double> ForceJacobianColumns::compute_centre(const Vec12& q,
                                                          const GlovePose& centre,
                                                          const std::vector<int>& columns,
                                                          double step_rad) {
    retarget_.reset();
    retarget_.set_initial_hint(nullptr);
    const Vec20 hint = retarget_.solve(centre.points_m);
    retarget_.set_initial_hint(&hint);

    std::vector<double> result(20 * columns.size());
    for (std::size_t k = 0; k < columns.size(); ++k) {
        Vec12 plus = q, minus = q;
        plus[columns[k]] += step_rad;
        minus[columns[k]] -= step_rad;
        const auto plus_pose = model_.evaluate_joints(plus, &centre);
        const auto minus_pose = model_.evaluate_joints(minus, &centre);
        retarget_.reset();
        const Vec20 plus_hand = retarget_.solve(plus_pose.points_m);
        retarget_.reset();
        const Vec20 minus_hand = retarget_.solve(minus_pose.points_m);
        for (std::size_t row = 0; row < 20; ++row) {
            const double value = (plus_hand[row] - minus_hand[row]) / (2.0 * step_rad);
            if (!std::isfinite(value)) throw std::runtime_error("non-finite retarget Jacobian");
            result[row * columns.size() + k] = value;
        }
    }
    // A failed calculation must not advance the CAD seed.
    seed_ = centre;
    return result;
}

}  // namespace motorhand::native
