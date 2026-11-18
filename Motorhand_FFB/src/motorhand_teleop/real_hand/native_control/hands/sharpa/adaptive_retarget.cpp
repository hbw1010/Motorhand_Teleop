#include "adaptive_retarget.hpp"

#include <nlopt.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace motorhand::native::sharpa {
namespace {
using V3 = Eigen::Vector3d;
using J3 = AdaptiveRetarget::Jacobian;
constexpr double eps = 1e-8;
constexpr std::array<const char*, 5> fingers = {"thumb", "index", "middle", "ring", "pinky"};

template <typename T>
T option(const YAML::Node& node, const char* key, T fallback) {
    return node && node[key] ? node[key].as<T>() : fallback;
}

double huber(double d, double delta) {
    return d <= delta ? 0.5 * d * d : delta * (d - 0.5 * delta);
}

double huber_grad(double d, double delta) { return std::min(d, delta); }

struct NloptDeleter { void operator()(nlopt_opt_s* p) const { if (p) nlopt_destroy(p); } };
using Opt = std::unique_ptr<nlopt_opt_s, NloptDeleter>;
void check(nlopt_result result, const char* operation) {
    if (result < 0) throw std::runtime_error(std::string("NLopt ") + operation + " failed: " + std::to_string(result));
}
}  // namespace

struct AdaptiveRetarget::Impl {
    Opt opt;
    Vector lower, upper, last, initial_hint;
    std::array<int, 5> pip_q, dip_q;
    EvaluateKinematics evaluate;
    Kinematics kinematics;
    bool has_last = false, has_initial_hint = false;
    double delta, delta_dir, norm_delta, w_pos, w_dir, w_full, scaling;
    double w_hyper, soft_min, w_couple, couple_ratio;
    bool thumb_skip;
    std::array<std::array<double, 3>, 5> segment{};
    std::array<double, 4> d1{}, d2{};
    double pinch_weight_max = 0.7;
    std::array<V3, 5> target_tip{}, target_dir{};
    std::array<std::array<V3, 3>, 5> target_full{};
    std::array<double, 5> alpha{};
    std::exception_ptr callback_error;

    Impl(const YAML::Node& ret, Vector lower_bounds, Vector upper_bounds,
         std::array<int, 5> pip_axes, std::array<int, 5> dip_axes, EvaluateKinematics callback)
        : lower(std::move(lower_bounds)), upper(std::move(upper_bounds)),
          last(Vector::Zero(lower.size())), initial_hint(Vector::Zero(lower.size())),
          pip_q(pip_axes), dip_q(dip_axes), evaluate(std::move(callback)) {
        const auto dimension = lower.size();
        if (!dimension || upper.size() != dimension || !lower.allFinite() || !upper.allFinite() ||
            (lower.array() > upper.array()).any() || !evaluate)
            throw std::invalid_argument("invalid retarget model bounds or kinematics");
        for (int i = 0; i < 5; ++i)
            for (const int axis : {pip_q[i], dip_q[i]})
                if (axis < 0 || axis >= dimension) throw std::invalid_argument("invalid retarget PIP/DIP axis");
        kinematics.wrist.jacobian_cm.setZero(3, dimension);
        for (int i = 0; i < 5; ++i)
            for (auto* point : {&kinematics.tip[i], &kinematics.pip[i], &kinematics.dip[i]})
                point->jacobian_cm.setZero(3, dimension);
        delta = option<double>(ret, "huber_delta", 2.0);
        delta_dir = option<double>(ret, "huber_delta_dir", 0.5);
        norm_delta = option<double>(ret, "norm_delta", 0.04);
        w_pos = option<double>(ret, "w_pos", 1.0);
        w_dir = option<double>(ret, "w_dir", 10.0);
        w_full = option<double>(ret, "w_full_hand", 1.0);
        scaling = option<double>(ret, "scaling", 1.0);
        w_hyper = option<double>(ret, "w_hyper", 0.0);
        soft_min = option<double>(ret, "soft_min", 0.0);
        w_couple = option<double>(ret, "w_couple", 0.0);
        couple_ratio = option<double>(ret, "couple_ratio", 0.7);
        thumb_skip = option<bool>(ret, "thumb_skip_pip", false);
        pinch_weight_max = option<double>(ret, "pinch_weight_max", 0.7);
        if (!std::isfinite(pinch_weight_max) || pinch_weight_max <= 0 || pinch_weight_max > 1)
            throw std::runtime_error("pinch_weight_max must be in (0, 1]");
        const auto scales = ret ? ret["segment_scaling"] : YAML::Node();
        const auto thresholds = ret ? ret["pinch_thresholds"] : YAML::Node();
        for (int i = 0; i < 5; ++i) {
            segment[i] = {1.0, 1.0, 1.0};
            if (scales && scales[fingers[i]]) {
                const auto values = scales[fingers[i]].as<std::vector<double>>();
                if (values.size() != 3 && values.size() != 4) throw std::runtime_error("invalid segment scaling");
                for (int k = 0; k < 3; ++k) segment[i][k] = values[k + (values.size() == 4 ? 1 : 0)];
            }
            if (i) {
                const auto t = thresholds ? thresholds[fingers[i]] : YAML::Node();
                d1[i-1] = option<double>(t, "d1", 2.0);
                d2[i-1] = option<double>(t, "d2", 4.0);
            }
        }
        opt.reset(nlopt_create(NLOPT_LD_SLSQP, dimension));
        if (!opt) throw std::runtime_error("cannot create NLopt optimizer");
        check(nlopt_set_lower_bounds(opt.get(), lower.data()), "lower bounds");
        check(nlopt_set_upper_bounds(opt.get(), upper.data()), "upper bounds");
        check(nlopt_set_maxeval(opt.get(), 50), "maxeval");
        check(nlopt_set_ftol_abs(opt.get(), 1e-4), "ftol");
        check(nlopt_set_min_objective(opt.get(), &objective, this), "objective");
    }

    static double objective(unsigned n, const double* x, double* grad, void* context) noexcept {
        auto& self = *static_cast<Impl*>(context);
        try {
            if (n != self.lower.size()) throw std::runtime_error("unexpected NLopt dimension");
            const Eigen::Map<const Vector> q(x, n);
            Vector g = Vector::Zero(n);
            const double result = self.loss(q, g);
            if (!std::isfinite(result) || !g.allFinite()) throw std::runtime_error("non-finite retarget objective");
            if (grad) {
                Eigen::Map<Vector> mapped_grad(grad, n);
                mapped_grad = g;
            }
            return result;
        } catch (...) {
            self.callback_error = std::current_exception();
            nlopt_force_stop(self.opt.get());
            return 0;
        }
    }

    double loss(const Vector& q, Vector& gradient) {
        evaluate(q, kinematics);
        const V3& wrist_p = kinematics.wrist.position_cm;
        const J3& wrist_j = kinematics.wrist.jacobian_cm;
        double total = 0;
        for (int i = 0; i < 5; ++i) {
            const V3& tp = kinematics.tip[i].position_cm;
            const V3& pp = kinematics.pip[i].position_cm;
            const V3& dp = kinematics.dip[i].position_cm;
            const J3& tj = kinematics.tip[i].jacobian_cm;
            const J3& pj = kinematics.pip[i].jacobian_cm;
            const J3& dj = kinematics.dip[i].jacobian_cm;
            auto vector_loss = [&](const V3& diff, const J3& jac, double weight, double threshold) {
                const double d = diff.norm();
                total += weight * huber(d, threshold);
                gradient.noalias() += weight * huber_grad(d, threshold) / (d + eps) * jac.transpose() * diff;
            };
            vector_loss(tp - wrist_p - target_tip[i], tj - wrist_j, alpha[i] * w_pos, delta);
            const V3 v = tp - dp;
            const double norm = v.norm();
            const V3 u = v / (norm + eps);
            const V3 diff = u - target_dir[i];
            const double dist = diff.norm();
            const double weight = alpha[i] * w_dir;
            total += weight * huber(dist, delta_dir);
            const Eigen::Matrix3d norm_j = (Eigen::Matrix3d::Identity() - u * u.transpose()) / (norm + eps);
            gradient.noalias() += weight * huber_grad(dist, delta_dir) / (dist + eps) * (tj - dj).transpose() * norm_j * diff;
            const double full_weight = (1.0 - alpha[i]) * w_full / ((i == 0 && thumb_skip) ? 2.0 : 3.0);
            if (!(i == 0 && thumb_skip)) vector_loss(pp - wrist_p - target_full[i][0], pj - wrist_j, full_weight, delta);
            vector_loss(dp - wrist_p - target_full[i][1], dj - wrist_j, full_weight, delta);
            vector_loss(tp - wrist_p - target_full[i][2], tj - wrist_j, full_weight, delta);
        }
        if (has_last) {
            total += norm_delta * (q - last).squaredNorm();
            gradient += 2.0 * norm_delta * (q - last);
        }
        for (int i = 0; i < 5; ++i) {
            for (int j : {pip_q[i], dip_q[i]}) {
                const double penalty = std::max(soft_min - q[j], 0.0);
                total += w_hyper * penalty * penalty;
                gradient[j] -= 2.0 * w_hyper * penalty;
            }
            const double diff = q[dip_q[i]] - couple_ratio * q[pip_q[i]];
            total += w_couple * diff * diff;
            gradient[dip_q[i]] += 2.0 * w_couple * diff;
            gradient[pip_q[i]] -= 2.0 * w_couple * couple_ratio * diff;
        }
        return total;
    }

    Vector solve(const ModelPoints& p) {
        for (const auto& point : p)
            if (!point.allFinite()) throw std::invalid_argument("non-finite canonical point");
        for (int i = 1; i < 5; ++i) {
            const double distance = 100.0 * (p[4 + 4*i] - p[4]).norm();
            alpha[i] = std::clamp((d2[i-1] - distance) / (d2[i-1] - d1[i-1] + eps), 0.0, pinch_weight_max);
        }
        alpha[0] = *std::max_element(alpha.begin() + 1, alpha.end());
        for (int i = 0; i < 5; ++i) {
            const int base = 4*i;
            target_tip[i] = 100.0 * scaling * (p[base+4] - p[0]);
            V3 direction = p[base+4] - p[base+3];
            target_dir[i] = direction / (direction.norm() + eps);
            for (int j = 0; j < 3; ++j) target_full[i][j] = 100.0 * segment[i][j] * (p[base+2+j] - p[0]);
        }
        Vector q(lower.size());
        for (int i = 0; i < lower.size(); ++i) {
            const double initial = has_initial_hint ? initial_hint[i] :
                has_last ? last[i] : (lower[i] + upper[i]) * 0.5;
            q[i] = std::clamp(initial, lower[i], upper[i]);
        }
        callback_error = nullptr;
        check(nlopt_set_force_stop(opt.get(), 0), "reset force stop");
        double cost = 0;
        const auto status = nlopt_optimize(opt.get(), q.data(), &cost);
        if (callback_error) std::rethrow_exception(callback_error);
        if (status <= 0) throw std::runtime_error("AdaptiveRetarget optimization failed: " + std::to_string(status));
        if (!std::isfinite(cost)) throw std::runtime_error("non-finite retarget result");
        Vector next(lower.size());
        for (int i = 0; i < lower.size(); ++i) {
            if (!std::isfinite(q[i])) throw std::runtime_error("non-finite retarget joint");
            // 保留原优化器的 float32 输出和下一帧热启动精度。
            next[i] = static_cast<double>(static_cast<float>(q[i]));
        }
        last = next;
        has_last = true;
        return next;
    }

};

AdaptiveRetarget::AdaptiveRetarget(const YAML::Node& retarget, Vector lower, Vector upper,
                                 std::array<int, 5> pip_axes, std::array<int, 5> dip_axes,
                                 EvaluateKinematics evaluate)
    : impl_(std::make_unique<Impl>(retarget, std::move(lower), std::move(upper),
                                  pip_axes, dip_axes, std::move(evaluate))) {}
AdaptiveRetarget::~AdaptiveRetarget() = default;
AdaptiveRetarget::AdaptiveRetarget(AdaptiveRetarget&&) noexcept = default;
AdaptiveRetarget& AdaptiveRetarget::operator=(AdaptiveRetarget&&) noexcept = default;
AdaptiveRetarget::Vector AdaptiveRetarget::solve(const ModelPoints& points_m) { return impl_->solve(points_m); }
std::array<double, 21> AdaptiveRetarget::task_scales() const {
    std::array<double, 21> scales;
    scales.fill(1.0);
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 3; ++j) scales[4 * i + 2 + j] = impl_->segment[i][j];
    return scales;
}
void AdaptiveRetarget::seed(const Vector& joint_rad) {
    if (joint_rad.size() != impl_->lower.size() || !joint_rad.allFinite())
        throw std::invalid_argument("invalid retarget seed");
    for (int i = 0; i < joint_rad.size(); ++i)
        impl_->last[i] = std::clamp(joint_rad[i], impl_->lower[i], impl_->upper[i]);
    impl_->has_last = true;
}
void AdaptiveRetarget::reset() { impl_->has_last = false; }
void AdaptiveRetarget::set_initial_hint(const Vector* hint) {
    if (!hint) {
        impl_->has_initial_hint = false;
        return;
    }
    if (hint->size() != impl_->lower.size() || !hint->allFinite())
        throw std::invalid_argument("invalid retarget initial hint");
    impl_->initial_hint = *hint;
    impl_->has_initial_hint = true;
}
void AdaptiveRetarget::configure_differentiation() {
    check(nlopt_set_ftol_abs(impl_->opt.get(), 1e-11), "differentiation ftol");
    check(nlopt_set_maxeval(impl_->opt.get(), 200), "differentiation maxeval");
}

}  // namespace motorhand::native::sharpa
