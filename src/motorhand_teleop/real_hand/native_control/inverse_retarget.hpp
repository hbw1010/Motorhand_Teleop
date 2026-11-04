#pragma once

#include "types.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>

namespace motorhand::native {

class Ve33Model;
class WujiRetarget;

struct InverseResult {
    GlovePose pose{};
    double task_error_max_mm{};
    double task_error_rms_mm{};
    int iterations{};
    bool success{};
    int optimizer_status{};  // NLopt status, or zero if optimization was not started.
    int optimizer_attempts{};
    std::array<int, 3> attempt_status{};
    // KKT fallback uses F=500*(sum(point_error_m^2)+1e-6*sum(dq_rad^2)).
    // Absolute thresholds: |delta F|<=1e-8, infinity-norm stationarity
    // <=1e-6 F/rad, complementarity <=1e-10 F, active slack <=1e-8 rad.
    // Feasibility is exact after CAD roundoff projection; no relative tolerance.
    bool kkt_converged{};
    double kkt_stationarity = std::numeric_limits<double>::infinity();
    double kkt_complementarity = std::numeric_limits<double>::infinity();
    double objective_change = std::numeric_limits<double>::infinity();
    std::string failure_reason;
};

class InverseRetarget {
public:
    InverseRetarget(Ve33Model& glove, WujiRetarget& hand,
                    std::array<std::array<double, 2>, 12> motor_bounds);
    ~InverseRetarget();
    InverseRetarget(const InverseRetarget&) = delete;
    InverseRetarget& operator=(const InverseRetarget&) = delete;
    InverseResult solve(const Vec20& actual_hand, const GlovePose& measured_seed);

private:
    struct Optimizer;
    std::unique_ptr<Optimizer> optimizer_;
    Ve33Model& glove_;
    WujiRetarget& hand_;
    std::array<std::array<double, 2>, 12> motor_bounds_;
    GlovePose warm_pose_{};
    Points warm_target_{};
    Vec12 warm_anchor_{};
    bool has_warm_pose_ = false;
};

}  // namespace motorhand::native
