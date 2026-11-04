#include "../src/motorhand_teleop/real_hand/native_control/linear_force.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace motorhand::native;
namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("linear force test failed"); }
void near(double a, double b) { require(std::abs(a - b) < 1e-12); }
template <typename F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw);
}
void tests() {
    LinearForceParameters config;
    config.wuji_max_current_a.fill(2);
    config.wuji_current_bias_a.fill(0.5);
    config.wuji_effort_sign.fill(1);
    config.wuji_effort_sign[1] = -1;
    config.gain = 0.5;
    Vec20 effort; effort.fill(0.5);
    Vec12 raw, sign, scale; raw.fill(2); sign.fill(1); scale.fill(1);
    std::array<int, 12> binding{};
    for (int i = 0; i < 12; ++i) binding[i] = i + 1;
    std::swap(binding[0], binding[1]); sign[0] = -1; scale[0] = 2;
    const Vec12 limits = linear_model_torque_limits(raw, sign, scale, binding);
    near(limits[0], 2); near(limits[1], 1);
    MotorJacobian motor{};
    for (int i = 0; i < 12; ++i) motor[i * 12 + i] = 1;
    motor[0] = 2;
    std::array<double, 240> retarget{};
    retarget[0] = 2; retarget[12] = -6;
    retarget[1] = 1;
    // Python project_max_torque_linear sample: B row 0 = [1,-3],
    // row 1 = [1,0], all remaining rows zero. Values below are exact
    // analytic fixtures for sample-by-sample comparison with that function.
    const double samples[][4] = {{2.5, -1.5, 0.5, -1},
                                 {1.5, 1.5, -0.5, -0.5},
                                 {100, -100, 0.5, -1},
                                 {0.5, 0.5, 0, 0}};
    for (const auto& sample : samples) {
        effort[0] = sample[0]; effort[1] = sample[1];
        const auto out = project_max_torque_linear(retarget, motor, effort, config, limits);
        near(out.model_torque_nm[0], sample[2]);
        near(out.model_torque_nm[1], sample[3] * 0.5);
        for (int i = 2; i < 12; ++i) near(out.model_torque_nm[i], 0);
    }
    effort[0] = std::numeric_limits<double>::max();
    effort[1] = -std::numeric_limits<double>::max();
    const auto saturated = project_max_torque_linear(retarget, motor, effort, config, limits);
    near(saturated.wuji_load_fraction[0], 1); near(saturated.wuji_load_fraction[1], 1);
    auto bad = motor; bad[0] = 0;
    rejects([&] { project_max_torque_linear(retarget, bad, effort, config, limits); });
    bad = motor; bad[0] = 1e-18;
    rejects([&] { project_max_torque_linear(retarget, bad, effort, config, limits); });
    auto invalid = config; invalid.wuji_max_current_a[0] = 0;
    rejects([&] { project_max_torque_linear(retarget, motor, effort, invalid, limits); });
    invalid = config; invalid.wuji_effort_sign[0] = 0;
    rejects([&] { project_max_torque_linear(retarget, motor, effort, invalid, limits); });
    invalid = config; invalid.gain = 1.1;
    rejects([&] { project_max_torque_linear(retarget, motor, effort, invalid, limits); });
    auto nan_effort = effort; nan_effort[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { project_max_torque_linear(retarget, motor, nan_effort, config, limits); });
    auto bad_binding = binding; bad_binding[0] = bad_binding[1];
    rejects([&] { linear_model_torque_limits(raw, sign, scale, bad_binding); });
    auto bad_scale = scale; bad_scale[0] = 0;
    rejects([&] { linear_model_torque_limits(raw, sign, bad_scale, binding); });
}
}
int main() { tests(); }
