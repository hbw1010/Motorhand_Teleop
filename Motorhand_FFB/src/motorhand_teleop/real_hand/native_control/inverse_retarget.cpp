#include "inverse_retarget.hpp"

#include "ve33_model.hpp"
#include "wuji_retarget.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <nlopt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace motorhand::native {
namespace {

constexpr double kJointBound = 1.5;
// This is an internal compute cap, not a replacement for source-age and
// command-deadline checks at the caller.
constexpr double kComputeBudgetSeconds = 0.070;
constexpr unsigned kMaxAttempts = 3;
constexpr double kRegularization = 1e-6;
constexpr double kMotorRoundoff = 1e-7;
constexpr unsigned kJoints = 12;
constexpr unsigned kConstraints = 24;

using Bounds = std::array<std::array<double, 2>, 12>;

template <typename T>
bool finite_array(const T& values) {
    for (double value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

bool finite_pose(const GlovePose& pose) {
    return finite_array(pose.joint_rad) && finite_array(pose.motor_rad) &&
           finite_array(pose.tree_rad) && finite_array(pose.points_m) &&
           finite_array(pose.point_jacobian) && finite_array(pose.motor_jacobian) &&
           std::isfinite(pose.max_gap_m);
}

// The objective and all motor constraints share a CAD evaluation at each q.
// A failed closure is infeasible, not a successful synthetic pose.
struct Evaluation {
    Ve33Model& model;
    const Points& target;
    const Vec12& anchor;
    const Bounds& motor_bounds;
    GlovePose seed;
    GlovePose pose{};
    Vec12 last_q{};
    bool cached = false;
    bool valid = false;
    int evaluations = 0;
    std::array<char, 256> last_error{};

    bool evaluate(const double* q) noexcept {
        if (cached && std::equal(last_q.begin(), last_q.end(), q)) return valid;
        std::copy_n(q, kJoints, last_q.begin());
        cached = true;
        valid = false;
        ++evaluations;
        if (!finite_array(last_q)) {
            std::snprintf(last_error.data(), last_error.size(), "non-finite optimizer joint candidate");
            return false;
        }
        try {
            GlovePose candidate = model.evaluate_joints(last_q, &seed);
            if (!finite_pose(candidate)) {
                std::snprintf(last_error.data(), last_error.size(), "non-finite CAD candidate or derivatives");
                return false;
            }
            pose = candidate;
            seed = candidate;
            valid = true;
        } catch (const std::exception& error) {
            std::snprintf(last_error.data(), last_error.size(), "%s", error.what());
        } catch (...) {
            std::snprintf(last_error.data(), last_error.size(), "unknown CAD exception");
        }
        return valid;
    }
};

double objective(unsigned, const double* q, double* gradient, void* data) noexcept {
    auto& eval = *static_cast<Evaluation*>(data);
    if (!eval.evaluate(q)) {
        if (gradient) std::fill_n(gradient, kJoints, 0.0);
        return 1e30;
    }
    double sum = 0.0;
    if (gradient) std::fill_n(gradient, kJoints, 0.0);
    for (unsigned row = 0; row < 63; ++row) {
        const double residual = eval.pose.points_m[row] - eval.target[row];
        sum += residual * residual;
        if (gradient) {
            for (unsigned col = 0; col < kJoints; ++col) {
                gradient[col] += 1000.0 * residual * eval.pose.point_jacobian[row * kJoints + col];
            }
        }
    }
    for (unsigned col = 0; col < kJoints; ++col) {
        const double delta = q[col] - eval.anchor[col];
        sum += kRegularization * delta * delta;
        if (gradient) gradient[col] += 1000.0 * kRegularization * delta;
    }
    return 500.0 * sum;
}

void constraints(unsigned, double* result, unsigned, const double* q,
                 double* gradient, void* data) noexcept {
    auto& eval = *static_cast<Evaluation*>(data);
    if (!eval.evaluate(q)) {
        std::fill_n(result, kConstraints, 10.0);
        if (gradient) std::fill_n(gradient, kConstraints * kJoints, 0.0);
        return;
    }
    for (unsigned motor = 0; motor < kJoints; ++motor) {
        const double value = eval.pose.motor_rad[motor];
        // NLopt's inequality convention is constraint <= 0.
        result[motor] = eval.motor_bounds[motor][0] - value;
        result[kJoints + motor] = value - eval.motor_bounds[motor][1];
        if (gradient) {
            for (unsigned col = 0; col < kJoints; ++col) {
                const double derivative = eval.pose.motor_jacobian[motor * kJoints + col];
                gradient[motor * kJoints + col] = -derivative;
                gradient[(kJoints + motor) * kJoints + col] = derivative;
            }
        }
    }
}


bool inside_bounds(const GlovePose& pose, const Bounds& motors) {
    for (unsigned i = 0; i < kJoints; ++i) {
        if (std::abs(pose.joint_rad[i]) > kJointBound ||
            pose.motor_rad[i] < motors[i][0] || pose.motor_rad[i] > motors[i][1]) return false;
    }
    return true;
}

// Find nonnegative multipliers minimizing the infinity-relevant stationarity
// residual using coordinate descent on ||grad(F) + A*lambda||_2^2.
// A contains only nearly active inequalities c(q)<=0; all remaining
// multipliers are zero. Failure to solve this NNLS tightly is a rejection.
void kkt_residual(Evaluation& eval, const Vec12& q, const Vec12& lower,
                  const Vec12& upper, double& stationarity, double& complementarity) {
    using Vector = Eigen::Matrix<double, 12, 1>;
    constexpr double active_slack = 1e-8;  // rad
    std::array<Vector, 48> columns;
    std::array<double, 48> slacks{};
    std::array<double, 48> multipliers{};
    unsigned count = 0;
    auto add = [&](const Vector& direction, double slack) {
        if (slack >= 0.0 && slack <= active_slack && direction.squaredNorm() > 1e-24) {
            columns[count] = direction;
            slacks[count++] = slack;
        }
    };
    for (unsigned i = 0; i < kJoints; ++i) {
        Vector axis = Vector::Zero();
        axis[i] = 1.0;
        add(-axis, q[i] - lower[i]);
        add(axis, upper[i] - q[i]);
        Vector motor;
        for (unsigned j = 0; j < kJoints; ++j)
            motor[j] = eval.pose.motor_jacobian[i * kJoints + j];
        add(-motor, eval.pose.motor_rad[i] - eval.motor_bounds[i][0]);
        add(motor, eval.motor_bounds[i][1] - eval.pose.motor_rad[i]);
    }
    Vec12 gradient{};
    objective(kJoints, q.data(), gradient.data(), &eval);
    if (!finite_array(gradient)) return;
    Vector residual = Eigen::Map<const Vector>(gradient.data());
    for (unsigned sweep = 0; sweep < 400; ++sweep) {
        double largest_change = 0.0;
        for (unsigned i = 0; i < count; ++i) {
            const double updated = std::max(0.0, multipliers[i] -
                columns[i].dot(residual) / columns[i].squaredNorm());
            const double change = updated - multipliers[i];
            residual += change * columns[i];
            multipliers[i] = updated;
            largest_change = std::max(largest_change, std::abs(change));
        }
        if (largest_change < 1e-14) break;
    }
    if (!residual.allFinite()) return;
    stationarity = residual.cwiseAbs().maxCoeff();
    complementarity = 0.0;
    for (unsigned i = 0; i < count; ++i)
        complementarity = std::max(complementarity, multipliers[i] * slacks[i]);
}

}  // namespace
struct InverseRetarget::Optimizer {
    nlopt_opt value = nlopt_create(NLOPT_LD_SLSQP, kJoints);
    ~Optimizer() { if (value) nlopt_destroy(value); }
};


InverseRetarget::InverseRetarget(Ve33Model& glove, WujiRetarget& hand, Bounds motor_bounds, bool scale_aware)
    : optimizer_(std::make_unique<Optimizer>()), glove_(glove), hand_(hand), motor_bounds_(motor_bounds) {
    scales_.fill(1.0);
    if (scale_aware) scales_ = hand_.task_scales();
    for (double scale : scales_)
        if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("retarget task scale must be positive");
    for (const auto& bounds : motor_bounds_) {
        if (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1]) || bounds[0] >= bounds[1]) {
            throw std::invalid_argument("motor bounds must be finite and strictly ordered");
        }
    }
}

InverseRetarget::~InverseRetarget() = default;

InverseResult InverseRetarget::solve(const Vec20& actual_hand, const GlovePose& measured_seed) {
    const auto started = std::chrono::steady_clock::now();
    InverseResult result;
    result.pose = measured_seed;
    result.task_error_max_mm = std::numeric_limits<double>::infinity();
    result.task_error_rms_mm = std::numeric_limits<double>::infinity();
    auto fail = [&](std::string reason) {
        result.failure_reason = std::move(reason);
        return result;
    };
    if (!finite_array(actual_hand)) return fail("non-finite Hand2 feedback");
    if (!finite_pose(measured_seed)) return fail("non-finite measured glove pose or derivatives");

    try {
        Points target = hand_.hand_points(actual_hand);
        // Undo the forward solve's per-point scaling (points are wrist-relative).
        for (int point = 0; point < 21; ++point)
            for (int axis = 0; axis < 3; ++axis) target[3 * point + axis] /= scales_[point];
        if (!finite_array(target)) return fail("Hand2 forward kinematics produced non-finite points");
        const GlovePose& initial = has_warm_pose_ ? warm_pose_ : measured_seed;
        const Bounds active = glove_.active_joint_bounds(initial);
        Vec12 lower{}, upper{}, q{};
        for (unsigned i = 0; i < kJoints; ++i) {
            if (std::isnan(active[i][0]) || std::isnan(active[i][1]))
                return fail("non-finite active CAD bound at joint " + std::to_string(i));
            lower[i] = std::max(-kJointBound, active[i][0]);
            upper[i] = std::min(kJointBound, active[i][1]);
            if (!(lower[i] < upper[i]))
                return fail("empty active CAD/joint bound intersection at joint " + std::to_string(i));
            q[i] = std::clamp(initial.joint_rad[i], lower[i], upper[i]);
        }
        Evaluation eval{glove_, target, measured_seed.joint_rad, motor_bounds_, initial};
        if (std::equal(q.begin(), q.end(), initial.joint_rad.begin())) {
            eval.last_q = q;
            eval.pose = initial;
            eval.cached = true;
            eval.valid = true;
        }

        // Predict the effect of target and measured-anchor changes using the
        // previous successful analytic Jacobian. This changes only the initial
        // guess: SLSQP still minimizes the original constrained objective.
        if (has_warm_pose_ && eval.valid) {
            using Matrix = Eigen::Matrix<double, 63, 12, Eigen::RowMajor>;
            using Vector = Eigen::Matrix<double, 12, 1>;
            const Eigen::Map<const Matrix> jac(warm_pose_.point_jacobian.data());
            const Eigen::Map<const Eigen::Matrix<double, 63, 1>> now(target.data());
            const Eigen::Map<const Eigen::Matrix<double, 63, 1>> before(warm_target_.data());
            const Eigen::Map<const Vector> anchor(measured_seed.joint_rad.data());
            const Eigen::Map<const Vector> old_anchor(warm_anchor_.data());
            Eigen::Matrix<double, 12, 12> normal = jac.transpose() * jac;
            normal.diagonal().array() += kRegularization;
            const Vector step = normal.ldlt().solve(
                jac.transpose() * (now - before) + kRegularization * (anchor - old_anchor));
            if (step.allFinite()) {
                Vec12 predicted = q;
                const double scale = std::max(1.0, step.cwiseAbs().maxCoeff() / 0.04);
                for (unsigned i = 0; i < kJoints; ++i)
                    predicted[i] = std::clamp(q[i] + step[i] / scale, lower[i], upper[i]);
                if (predicted != q && eval.evaluate(predicted.data()) &&
                    inside_bounds(eval.pose, motor_bounds_)) {
                    q = predicted;
                } else {
                    // A prediction is optional and cannot replace a valid
                    // previous branch with an infeasible or failed closure.
                    eval.seed = initial;
                    eval.pose = initial;
                    eval.last_q = q;
                    eval.cached = true;
                    eval.valid = true;
                }
            }
        }

        nlopt_opt opt = optimizer_->value;
        if (!opt) return fail("could not create NLopt SLSQP optimizer");
        std::array<double, kConstraints> tolerance{};
        nlopt_result setup = nlopt_remove_inequality_constraints(opt);
        if (setup >= 0) setup = nlopt_set_lower_bounds(opt, lower.data());
        if (setup >= 0) setup = nlopt_set_upper_bounds(opt, upper.data());
        if (setup >= 0) setup = nlopt_set_min_objective(opt, objective, &eval);
        if (setup >= 0) setup = nlopt_add_inequality_mconstraint(opt, kConstraints, constraints, &eval, tolerance.data());
        if (setup >= 0) setup = nlopt_set_ftol_abs(opt, 1e-8);
        if (setup >= 0) setup = nlopt_set_maxeval(opt, 80);
        if (setup < 0) {
            result.optimizer_status = static_cast<int>(setup);
            return fail("NLopt setup failed, status " + std::to_string(setup));
        }
        double minimum = 0.0;
        nlopt_result status = NLOPT_FAILURE;
        bool valid = false;
        bool converged = false;
        std::string retry_stop;
        double previous_feasible_objective = std::numeric_limits<double>::infinity();
        for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const double remaining = kComputeBudgetSeconds - elapsed;
            if (remaining <= 0.0) {
                retry_stop = "internal compute budget exhausted";
                break;
            }
            const nlopt_result time_setup = nlopt_set_maxtime(opt, remaining);
            if (time_setup < 0) {
                status = time_setup;
                retry_stop = "setting optimizer compute budget failed";
                break;
            }
            status = nlopt_optimize(opt, q.data(), &minimum);
            result.attempt_status[attempt] = static_cast<int>(status);
            ++result.optimizer_attempts;
            result.optimizer_status = static_cast<int>(status);
            // MAXEVAL, MAXTIME and ROUNDOFF_LIMITED do not mean convergence.
            converged = status == NLOPT_SUCCESS || status == NLOPT_STOPVAL_REACHED ||
                        status == NLOPT_FTOL_REACHED || status == NLOPT_XTOL_REACHED;
            valid = eval.evaluate(q.data());
            result.iterations = eval.evaluations;
            if (converged || (status != NLOPT_ROUNDOFF_LIMITED && status != NLOPT_MAXEVAL_REACHED))
                break;
            if (!valid || !std::isfinite(minimum)) {
                retry_stop = "restart candidate is not a finite CAD solution";
                break;
            }

            // Restart only from an independently validated feasible CAD pose.
            // Roundoff-sized motor excess is projected through the real closed
            // chain exactly as for the final result, never by mapping q to m.
            GlovePose candidate = eval.pose;
            Vec12 bounded = candidate.motor_rad;
            bool project = false;
            bool feasible = true;
            for (unsigned i = 0; i < kJoints; ++i) {
                if (candidate.joint_rad[i] < lower[i] || candidate.joint_rad[i] > upper[i])
                    feasible = false;
                bounded[i] = std::clamp(bounded[i], motor_bounds_[i][0], motor_bounds_[i][1]);
                const double excess = std::abs(bounded[i] - candidate.motor_rad[i]);
                if (excess > kMotorRoundoff) feasible = false;
                project = project || excess > 0.0;
            }
            if (!feasible) {
                retry_stop = "restart candidate violates joint or motor limits";
                break;
            }
            if (project) candidate = glove_.evaluate_motors(bounded, &candidate);
            if (!finite_pose(candidate) || !inside_bounds(candidate, motor_bounds_)) {
                retry_stop = "restart projection produced invalid or out-of-limit CAD pose";
                break;
            }
            for (unsigned i = 0; i < kJoints; ++i) {
                if (candidate.joint_rad[i] < lower[i] || candidate.joint_rad[i] > upper[i])
                    feasible = false;
            }
            if (!feasible) {
                retry_stop = "restart projection violates active joint bounds";
                break;
            }
            q = candidate.joint_rad;
            eval.seed = candidate;
            eval.pose = candidate;
            eval.last_q = q;
            eval.cached = true;
            eval.valid = true;
            const double feasible_objective = objective(kJoints, q.data(), nullptr, &eval);
            if (!std::isfinite(feasible_objective)) {
                retry_stop = "non-finite objective at feasible restart candidate";
                break;
            }
            if (std::isfinite(previous_feasible_objective))
                result.objective_change = std::abs(feasible_objective - previous_feasible_objective);
            previous_feasible_objective = feasible_objective;
            if ((status == NLOPT_ROUNDOFF_LIMITED || status == NLOPT_MAXEVAL_REACHED) &&
                std::isfinite(result.objective_change)) {
                kkt_residual(eval, q, lower, upper, result.kkt_stationarity,
                             result.kkt_complementarity);
                if (result.objective_change <= 1e-8 && result.kkt_stationarity <= 1e-6 &&
                    result.kkt_complementarity <= 1e-10) {
                    result.kkt_converged = true;
                    converged = true;
                    minimum = feasible_objective;
                    break;
                }
            }
            if (attempt + 1 == kMaxAttempts) {
                retry_stop = "restart limit reached";
                break;
            }
        }
        result.optimizer_status = static_cast<int>(status);
        if (!converged || !valid || !std::isfinite(minimum)) {
            std::string reason = "NLopt status " + std::to_string(status);
            if (!converged) reason += " (not converged)";
            if (!valid) reason += "; final CAD candidate invalid";
            if (!std::isfinite(minimum)) reason += "; non-finite objective";
            if (!retry_stop.empty()) reason += "; " + retry_stop;
            if (std::isfinite(result.objective_change))
                reason += "; objective_change=" + std::to_string(result.objective_change) +
                          "; stationarity=" + std::to_string(result.kkt_stationarity) +
                          "; complementarity=" + std::to_string(result.kkt_complementarity);
            if (eval.last_error[0]) reason += "; last CAD error: " + std::string(eval.last_error.data());
            if (const char* message = nlopt_get_errmsg(opt)) reason += "; " + std::string(message);
            return fail(std::move(reason));
        }

        GlovePose solved = eval.pose;
        Vec12 bounded = solved.motor_rad;
        bool project = false;
        for (unsigned i = 0; i < kJoints; ++i) {
            if (solved.joint_rad[i] < lower[i] || solved.joint_rad[i] > upper[i])
                return fail("solution outside active joint bounds at joint " + std::to_string(i));
            bounded[i] = std::clamp(bounded[i], motor_bounds_[i][0], motor_bounds_[i][1]);
            const double excess = std::abs(bounded[i] - solved.motor_rad[i]);
            if (excess > kMotorRoundoff)
                return fail("motor " + std::to_string(i) + " exceeds soft limit by " +
                            std::to_string(excess) + " rad");
            project = project || excess > 0.0;
        }
        if (project) solved = glove_.evaluate_motors(bounded, &solved);
        if (!finite_pose(solved)) return fail("projected CAD pose or derivatives are non-finite");
        if (!inside_bounds(solved, motor_bounds_))
            return fail("projected CAD pose violates joint or motor limits");

        double max_squared = 0.0;
        double sum_squared = 0.0;
        for (unsigned point = 0; point < 21; ++point) {
            double squared = 0.0;
            for (unsigned axis = 0; axis < 3; ++axis) {
                const double d = solved.points_m[point * 3 + axis] - target[point * 3 + axis];
                squared += d * d;
            }
            max_squared = std::max(max_squared, squared);
            sum_squared += squared;
        }
        result.task_error_max_mm = 1000.0 * std::sqrt(max_squared);
        result.task_error_rms_mm = 1000.0 * std::sqrt(sum_squared / 21.0);
        if (!std::isfinite(result.task_error_max_mm) || !std::isfinite(result.task_error_rms_mm))
            return fail("non-finite final task error");
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            > kComputeBudgetSeconds)
            return fail("internal compute budget exhausted before result publication");
        result.pose = solved;
        result.success = true;
        warm_pose_ = solved;
        warm_target_ = target;
        warm_anchor_ = measured_seed.joint_rad;
        has_warm_pose_ = true;
    } catch (const std::exception& error) {
        return fail(std::string("inverse retarget exception: ") + error.what());
    } catch (...) {
        return fail("unknown inverse retarget exception");
    }
    return result;
}

}  // namespace motorhand::native
