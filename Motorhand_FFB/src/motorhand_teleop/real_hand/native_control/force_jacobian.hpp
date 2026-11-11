#pragma once

#include "ve33_model.hpp"
#include "wuji_retarget.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace motorhand::native {

// One instance per worker thread; never shares the forward control optimizer.
class ForceJacobianColumns {
public:
    ForceJacobianColumns(const std::filesystem::path& asset_root,
                         const std::filesystem::path& yaml_path, bool left);

    // Returns a row-major 20 x columns.size() matrix in requested column order.
    // The caller supplies the configured finite-difference step in radians and
    // retains the source joint vector and source timestamp with the result.
    std::vector<double> compute(const Vec12& q, const std::vector<int>& columns,
                                double step_rad);
    // Revalidates the shared actual CAD tree and its joint vector, so every
    // worker differentiates the same physical assembly branch.
    std::vector<double> compute(const GlovePose& base_pose,
                                const std::vector<int>& columns, double step_rad);

private:
    std::vector<double> compute_centre(const Vec12& q, const GlovePose& centre,
                                       const std::vector<int>& columns, double step_rad);
    Ve33Model model_;
    WujiRetarget retarget_;
    std::optional<GlovePose> seed_;
};

}  // namespace motorhand::native
