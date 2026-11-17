#include "linear_force.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace motorhand::native {
namespace {
using Matrix12 = Eigen::Matrix<double, 12, 12, Eigen::RowMajor>;
using Matrix20x12 = Eigen::Matrix<double, 20, 12, Eigen::RowMajor>;
using Mapping = Eigen::Matrix<double, 12, 20>;

template <typename T> void require_finite(const T& values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("linear force: non-finite input");
}
}

Vec12 linear_model_torque_limits(const Vec12& raw_limits_nm, const Vec12& sign,
                                 const Vec12& scale, const std::array<int, 12>& raw_to_model) {
    Vec12 result{};
    std::array<bool, 12> seen{};
    for (int raw = 0; raw < 12; ++raw) {
        const int model = raw_to_model[raw] - 1;
        const double factor = sign[raw] * scale[raw];
        if (model < 0 || model >= 12 || seen[model] || !std::isfinite(raw_limits_nm[raw]) ||
            raw_limits_nm[raw] <= 0 || !std::isfinite(sign[raw]) || !std::isfinite(scale[raw]) ||
            !std::isfinite(factor) || factor == 0)
            throw std::invalid_argument("linear force: invalid raw motor binding or limit");
        seen[model] = true;
        result[model] = raw_limits_nm[raw] / std::abs(factor);
        if (!std::isfinite(result[model]) || result[model] <= 0)
            throw std::invalid_argument("linear force: invalid model motor limit");
    }
    return result;
}

LinearForceProjection project_max_torque_linear(
    const std::array<double, 240>& retarget, const MotorJacobian& motor,
    const Vec20& effort_a, const LinearForceParameters& config, const Vec12& model_torque_limits_nm) {
    require_finite(retarget); require_finite(motor); require_finite(effort_a);
    require_finite(config.wuji_max_current_a); require_finite(config.wuji_current_bias_a);
    require_finite(config.wuji_effort_sign); require_finite(model_torque_limits_nm);
    if (!std::isfinite(config.gain) || config.gain <= 0 || config.gain > 1)
        throw std::invalid_argument("linear force: invalid gain");
    LinearForceProjection result;
    for (int i = 0; i < 20; ++i) {
        if (config.wuji_max_current_a[i] <= 0 ||
            (config.wuji_effort_sign[i] != 1 && config.wuji_effort_sign[i] != -1))
            throw std::invalid_argument("linear force: invalid current configuration");
        // As in numpy, finite overrange currents may overflow to infinity and
        // are clipped; NaN is never a valid load.
        const double value = (effort_a[i] - config.wuji_current_bias_a[i]) *
                             config.wuji_effort_sign[i] / config.wuji_max_current_a[i];
        if (std::isnan(value)) throw std::invalid_argument("linear force: invalid load");
        result.wuji_load_fraction[i] = std::clamp(value, -1.0, 1.0);
    }
    for (double limit : model_torque_limits_nm)
        if (limit <= 0) throw std::invalid_argument("linear force: invalid model limit");

    const Eigen::Map<const Matrix12> motor_matrix(motor.data());
    const Eigen::Map<const Matrix20x12> retarget_matrix(retarget.data());
    // numpy.linalg.matrix_rank uses largest singular value * max(shape) * eps.
    Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd(motor_matrix);
    const auto singular = svd.singularValues();
    if (!singular.allFinite() || singular[11] <= singular[0] * 12 * std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("linear force: singular motor Jacobian");
    Eigen::PartialPivLU<Eigen::Matrix<double, 12, 12>> lu(motor_matrix.transpose());
    const Mapping mapping = lu.solve(retarget_matrix.transpose());
    if (!mapping.allFinite()) throw std::invalid_argument("linear force: non-finite mapping");
    for (int row = 0; row < 12; ++row) {
        double capacity = 0, reflected = 0;
        for (int col = 0; col < 20; ++col) {
            capacity += std::abs(mapping(row, col));
            reflected += mapping(row, col) * result.wuji_load_fraction[col];
        }
        if (!std::isfinite(capacity) || !std::isfinite(reflected))
            throw std::invalid_argument("linear force: non-finite projection");
        const double normalized = capacity == 0 ? 0 : reflected / capacity;
        result.model_torque_nm[row] = model_torque_limits_nm[row] *
            std::clamp(-config.gain * normalized, -1.0, 1.0);
        if (!std::isfinite(result.model_torque_nm[row]))
            throw std::invalid_argument("linear force: non-finite torque");
    }
    return result;
}
} // namespace motorhand::native
