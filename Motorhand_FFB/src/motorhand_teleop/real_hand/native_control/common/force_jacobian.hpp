#pragma once
#include "common/hand.hpp"

#include "ve33_model.hpp"
#include "common/hand.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace motorhand::native {

// One instance per worker thread; never shares the forward control optimizer.
template<std::size_t N> class CommonForceJacobianColumns {
public:
    CommonForceJacobianColumns(const std::filesystem::path& asset_root,
                         HandModelFactory<N> hand_factory, bool left);

    // Returns a row-major N x columns.size() matrix in requested column order.
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
    std::unique_ptr<HandModel<N>> retarget_;
    std::optional<GlovePose> seed_;
};

}  // namespace motorhand::native
