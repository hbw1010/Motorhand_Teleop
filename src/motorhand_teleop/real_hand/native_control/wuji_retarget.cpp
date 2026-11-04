#include "wuji_retarget.hpp"

#include <Eigen/Core>
#include <nlopt.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace motorhand::native {
namespace {
using V3 = Eigen::Vector3d;
using V20 = Eigen::Matrix<double, 20, 1>;
using J3 = Eigen::Matrix<double, 3, 20>;
using J6 = Eigen::Matrix<double, 6, 20>;
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

std::string expand(std::string pattern, const std::string& finger) {
    std::size_t pos = 0;
    while ((pos = pattern.find("{finger}", pos)) != std::string::npos) {
        pattern.replace(pos, 8, finger);
        pos += finger.size();
    }
    return pattern;
}

struct NloptDeleter { void operator()(nlopt_opt_s* p) const { if (p) nlopt_destroy(p); } };
using Opt = std::unique_ptr<nlopt_opt_s, NloptDeleter>;
void check(nlopt_result result, const char* operation) {
    if (result < 0) throw std::runtime_error(std::string("NLopt ") + operation + " failed: " + std::to_string(result));
}
}  // namespace

struct WujiRetarget::Impl {
    pinocchio::Model model;
    std::unique_ptr<pinocchio::Data> data;
    Opt opt;
    std::array<int, 20> semantic{};
    pinocchio::FrameIndex wrist{};
    std::array<pinocchio::FrameIndex, 5> tip{}, pip{}, dip{}, link1{};
    std::array<int, 5> pip_q{}, dip_q{};
    std::array<double, 20> lower{}, upper{};
    V20 last = V20::Zero();
    bool has_last = false;
    V20 initial_hint = V20::Zero();
    bool has_initial_hint = false;
    double delta, delta_dir, norm_delta, w_pos, w_dir, w_full, scaling;
    double w_hyper, soft_min, w_couple, couple_ratio;
    bool thumb_skip;
    std::array<std::array<double, 3>, 5> segment{};
    std::array<double, 4> d1{}, d2{};
    std::array<V3, 5> target_tip{}, target_dir{};
    std::array<std::array<V3, 3>, 5> target_full{};
    std::array<double, 5> alpha{};
    std::exception_ptr callback_error;

    Impl(const std::filesystem::path& yaml_path, bool left) {
        const auto root = YAML::LoadFile(yaml_path.string());
        const auto config = root["optimizer"];
        const auto ret = root["retarget"];
        if (!config || !config["urdf_path"])
            throw std::runtime_error("Hand2 retarget config requires urdf_path");
        if (option<std::string>(config, "type", "AdaptiveOptimizerAnalytical") != "AdaptiveOptimizerAnalytical")
            throw std::runtime_error("unsupported retarget optimizer type");
        auto urdf = std::filesystem::path(config["urdf_path"].as<std::string>());
        if (urdf.is_relative()) urdf = std::filesystem::absolute(yaml_path).parent_path() / urdf;
        pinocchio::urdf::buildModel(urdf.lexically_normal().string(), model);
        if (model.nq != 20 || model.nv != 20)
            throw std::runtime_error("Hand2 URDF must have exactly 20 scalar joints");
        data = std::make_unique<pinocchio::Data>(model);
        const std::string side = left ? "l_" : "r_";
        std::vector<std::string> names = {side + "thumb_cmc_flex", side + "thumb_cmc_abd", side + "thumb_mcp", side + "thumb_ip"};
        for (int i = 1; i < 5; ++i) {
            const std::string prefix = side + fingers[i] + (i == 4 ? "_" : "_finger_");
            for (const char* suffix : {"mcp_flex", "mcp_abd", "pip", "dip"}) names.push_back(prefix + suffix);
        }
        std::set<int> indices;
        for (int i = 0; i < 20; ++i) {
            if (!model.existJointName(names[i])) throw std::runtime_error("missing Hand2 joint: " + names[i]);
            const auto id = model.getJointId(names[i]);
            const auto& joint = model.joints[id];
            if (joint.nq() != 1 || joint.nv() != 1 || joint.idx_q() != joint.idx_v())
                throw std::runtime_error("Hand2 joint is not scalar: " + names[i]);
            semantic[i] = joint.idx_q();
            indices.insert(semantic[i]);
        }
        if (indices.size() != 20) throw std::runtime_error("duplicate Hand2 joint indices");
        for (int i = 0; i < 20; ++i) {
            lower[i] = model.lowerPositionLimit[i];
            upper[i] = model.upperPositionLimit[i];
            if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i] > upper[i])
                throw std::runtime_error("invalid Hand2 URDF limits");
        }
        const auto naming = config["link_naming"];
        const auto prefix = option<std::string>(naming, "prefix", "");
        const auto finger_names = option<std::vector<std::string>>(
            naming, "fingers", {"finger1", "finger2", "finger3", "finger4", "finger5"});
        if (finger_names.size() != 5) throw std::runtime_error("link_naming.fingers must have five entries");
        auto frame = [&](const std::string& name) {
            auto id = model.getFrameId(name, pinocchio::BODY);
            if (id >= model.nframes)
                id = model.getFrameId(std::string(left ? "left_" : "right_") + name, pinocchio::BODY);
            if (id >= model.nframes) throw std::runtime_error("missing Hand2 body frame: " + name);
            return id;
        };
        wrist = frame(prefix + option<std::string>(naming, "palm", "palm_link"));
        std::set<int> flex_indices;
        for (int i = 0; i < 5; ++i) {
            auto get = [&](const char* key, const char* fallback) {
                return frame(prefix + expand(option<std::string>(naming, key, fallback), finger_names[i]));
            };
            tip[i] = get("tip", "{finger}_tip_link");
            pip[i] = get("pip", "{finger}_link3");
            dip[i] = get("dip", "{finger}_link4");
            link1[i] = get("link1", "{finger}_link1");
            auto parent_index = [&](pinocchio::FrameIndex f) {
                const auto& j = model.joints[model.frames[f].parentJoint];
                if (j.nq() != 1 || j.nv() != 1) throw std::runtime_error("PIP/DIP frame has non-scalar parent");
                return j.idx_q();
            };
            pip_q[i] = parent_index(pip[i]); dip_q[i] = parent_index(dip[i]);
            flex_indices.insert(pip_q[i]); flex_indices.insert(dip_q[i]);
        }
        if (flex_indices.size() != 10) throw std::runtime_error("duplicate PIP/DIP parent joints");
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
        const auto fixed = ret ? ret["fixed_joints"] : YAML::Node();
        if (fixed && !fixed.IsNull()) {
            if (!fixed.IsMap()) throw std::runtime_error("fixed_joints must be a mapping");
            for (const auto& entry : fixed) {
                const auto name = entry.first.as<std::string>();
                const auto it = std::find(names.begin(), names.end(), name);
                if (it == names.end()) throw std::runtime_error("unknown fixed joint: " + name);
                const int index = semantic[std::distance(names.begin(), it)];
                const double value = entry.second.as<double>();
                if (!std::isfinite(value) || value < lower[index] || value > upper[index])
                    throw std::runtime_error("invalid fixed joint value: " + name);
                lower[index] = upper[index] = value;
            }
        }
        // The pipeline deliberately does not estimate these four axes.
        for (int i : {5, 9, 13, 17}) {
            const int j = semantic[i];
            if (lower[j] > 0 || upper[j] < 0) throw std::runtime_error("abduction cannot be fixed at zero");
            lower[j] = upper[j] = 0;
        }
        opt.reset(nlopt_create(NLOPT_LD_SLSQP, 20));
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
            if (n != 20) throw std::runtime_error("unexpected NLopt dimension");
            Eigen::Map<const V20> q(x);
            V20 g = V20::Zero();
            const double result = self.loss(q, g);
            if (!std::isfinite(result) || !g.allFinite()) throw std::runtime_error("non-finite retarget objective");
            if (grad) {
                Eigen::Map<V20> mapped_grad(grad);
                mapped_grad = g;
            }
            return result;
        } catch (...) {
            self.callback_error = std::current_exception();
            nlopt_force_stop(self.opt.get());
            return 0;
        }
    }

    double loss(const V20& q, V20& gradient) {
        pinocchio::computeJointJacobians(model, *data, q);
        pinocchio::updateFramePlacements(model, *data);
        auto position = [&](pinocchio::FrameIndex f) -> V3 { return 100.0 * data->oMf[f].translation(); };
        auto jacobian = [&](pinocchio::FrameIndex f) -> J3 {
            J6 j = J6::Zero();
            pinocchio::getFrameJacobian(model, *data, f, pinocchio::LOCAL_WORLD_ALIGNED, j);
            return 100.0 * j.topRows<3>();
        };
        const V3 wrist_p = position(wrist);
        const J3 wrist_j = jacobian(wrist);
        double total = 0;
        for (int i = 0; i < 5; ++i) {
            const V3 tp = position(tip[i]), pp = position(pip[i]), dp = position(dip[i]);
            const J3 tj = jacobian(tip[i]), pj = jacobian(pip[i]), dj = jacobian(dip[i]);
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

    Vec20 solve(const Points& points) {
        std::array<V3, 21> p;
        for (int i = 0; i < 21; ++i) {
            const double x = points[3*i], y = points[3*i+1], z = points[3*i+2];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) throw std::invalid_argument("non-finite canonical point");
            p[i] = V3(y, -x, z);
        }
        for (int i = 1; i < 5; ++i) {
            const double distance = 100.0 * (p[4 + 4*i] - p[4]).norm();
            alpha[i] = std::clamp((d2[i-1] - distance) / (d2[i-1] - d1[i-1] + eps), 0.0, 0.7);
        }
        alpha[0] = *std::max_element(alpha.begin() + 1, alpha.end());
        for (int i = 0; i < 5; ++i) {
            const int base = 4*i;
            target_tip[i] = 100.0 * scaling * (p[base+4] - p[0]);
            V3 direction = p[base+4] - p[base+3];
            target_dir[i] = direction / (direction.norm() + eps);
            for (int j = 0; j < 3; ++j) target_full[i][j] = 100.0 * segment[i][j] * (p[base+2+j] - p[0]);
        }
        std::array<double, 20> q;
        for (int i = 0; i < 20; ++i) {
            const double initial = has_initial_hint ? initial_hint[i] :
                has_last ? last[i] : (lower[i] + upper[i]) * 0.5;
            q[i] = std::clamp(initial, lower[i], upper[i]);
        }
        callback_error = nullptr;
        check(nlopt_set_force_stop(opt.get(), 0), "reset force stop");
        double cost = 0;
        const auto status = nlopt_optimize(opt.get(), q.data(), &cost);
        if (callback_error) std::rethrow_exception(callback_error);
        if (status <= 0) throw std::runtime_error("WujiRetarget optimization failed: " + std::to_string(status));
        if (!std::isfinite(cost)) throw std::runtime_error("non-finite retarget result");
        V20 next;
        for (int i = 0; i < 20; ++i) {
            if (!std::isfinite(q[i])) throw std::runtime_error("non-finite retarget joint");
            // The Python optimizer returns float32 and stores that value as the next warm start.
            next[i] = static_cast<double>(static_cast<float>(q[i]));
        }
        Vec20 output{};
        for (int i = 0; i < 20; ++i) output[i] = next[semantic[i]];
        last = next;
        has_last = true;
        return output;
    }

    Points hand_points(const Vec20& actual) {
        V20 q = V20::Zero();
        for (int i = 0; i < 20; ++i) {
            if (!std::isfinite(actual[i])) throw std::invalid_argument("non-finite Hand2 joint");
            q[semantic[i]] = actual[i];
        }
        pinocchio::forwardKinematics(model, *data, q);
        pinocchio::updateFramePlacements(model, *data);
        const V3 origin = data->oMf[wrist].translation();
        Points out{};
        for (int i = 0; i < 5; ++i) {
            const std::array<pinocchio::FrameIndex, 4> frames = {link1[i], pip[i], dip[i], tip[i]};
            for (int j = 0; j < 4; ++j) {
                const V3 v = data->oMf[frames[j]].translation() - origin;
                const int k = 3 * (1 + 4*i + j);
                out[k] = -v.y(); out[k+1] = v.x(); out[k+2] = v.z();
            }
        }
        return out;
    }
};

WujiRetarget::WujiRetarget(const std::filesystem::path& yaml_path, bool left)
    : impl_(std::make_unique<Impl>(yaml_path, left)) {}
WujiRetarget::~WujiRetarget() = default;
WujiRetarget::WujiRetarget(WujiRetarget&&) noexcept = default;
WujiRetarget& WujiRetarget::operator=(WujiRetarget&&) noexcept = default;
Vec20 WujiRetarget::solve(const Points& points) { return impl_->solve(points); }
Points WujiRetarget::hand_points(const Vec20& actual) { return impl_->hand_points(actual); }
void WujiRetarget::reset() { impl_->has_last = false; }
void WujiRetarget::configure_differentiation() {
    check(nlopt_set_ftol_abs(impl_->opt.get(), 1e-11), "differentiation ftol");
    check(nlopt_set_maxeval(impl_->opt.get(), 200), "differentiation maxeval");
}
void WujiRetarget::set_initial_hint(const Vec20* hint) {
    if (!hint) {
        impl_->has_initial_hint = false;
        return;
    }
    for (int i = 0; i < 20; ++i) {
        if (!std::isfinite((*hint)[i])) throw std::invalid_argument("non-finite retarget initial hint");
        impl_->initial_hint[impl_->semantic[i]] = (*hint)[i];
    }
    impl_->has_initial_hint = true;
}

}  // namespace motorhand::native
