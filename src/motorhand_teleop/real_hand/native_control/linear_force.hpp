#pragma once

#include "types.hpp"
#include <array>

namespace motorhand::native {

// These are the numeric fields of the explicitly acknowledged linear estimate
// configuration, not a current-to-torque calibration or measured contact force.
struct LinearForceParameters {
    Vec20 wuji_max_current_a{};
    Vec20 wuji_current_bias_a{};
    Vec20 wuji_effort_sign{};
    double gain{};
};

struct LinearForceProjection {
    Vec20 wuji_load_fraction{}; // dimensionless, clipped to [-1, 1]
    Vec12 model_torque_nm{};    // nominal model-motor-axis Nm, not measured force
};

// raw_to_model is a permutation of one-based CAD motor rows, indexed by raw
// SDK motor ID minus one. raw torque = model torque * sign * scale. Positive
// raw limits therefore become model limits divided by abs(sign * scale).
Vec12 linear_model_torque_limits(const Vec12& raw_limits_nm, const Vec12& sign,
                                 const Vec12& scale,
                                 const std::array<int, 12>& raw_to_model);

// Pure instantaneous projection: retarget is row-major 20x12, motor is
// row-major 12x12; limits are positive model-axis nominal Nm. No ramp, slew,
// freshness checks, timestamp manipulation, device output, or allocations on
// the successful path. Caller must enforce original source age and safety.
// Invalid inputs, rank-deficient Jacobians, or non-finite arithmetic throw.
LinearForceProjection project_max_torque_linear(
    const std::array<double, 240>& retarget, const MotorJacobian& motor,
    const Vec20& effort_a, const LinearForceParameters& config,
    const Vec12& model_torque_limits_nm);

} // namespace motorhand::native
