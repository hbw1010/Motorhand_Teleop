#pragma once

#include <Eigen/Core>

#include <array>
#include <functional>
#include <memory>

namespace YAML { class Node; }

namespace motorhand::native::sharpa {

// Sharpa私有自适应重定向，按Wuji目标函数实现；不修改或调用Wuji链路。
// 状态独立、单线程；所有关节角使用弧度。
class AdaptiveRetarget {
public:
    using Vector = Eigen::VectorXd;
    using Jacobian = Eigen::Matrix<double, 3, Eigen::Dynamic>;
    using ModelPoints = std::array<Eigen::Vector3d, 21>;
    struct TaskPoint {
        Eigen::Vector3d position_cm;
        Jacobian jacobian_cm;
    };
    struct Kinematics {
        TaskPoint wrist;
        std::array<TaskPoint, 5> tip, pip, dip;
    };
    // 回调须填充所有任务点；位置及其相对弧度的导数均以厘米计。
    // 求解器预先分配 3×N Jacobian，回调不能改变其维数。
    using EvaluateKinematics = std::function<void(const Vector&, Kinematics&)>;

    AdaptiveRetarget(const YAML::Node& retarget, Vector lower, Vector upper,
                     std::array<int, 5> pip_axes, std::array<int, 5> dip_axes,
                     EvaluateKinematics evaluate);
    ~AdaptiveRetarget();
    AdaptiveRetarget(AdaptiveRetarget&&) noexcept;
    AdaptiveRetarget& operator=(AdaptiveRetarget&&) noexcept;
    AdaptiveRetarget(const AdaptiveRetarget&) = delete;
    AdaptiveRetarget& operator=(const AdaptiveRetarget&) = delete;

    // 输入为模型坐标系下的 MediaPipe 21 点，单位米；输出按模型轴序。
    Vector solve(const ModelPoints& points_m);
    std::array<double, 21> task_scales() const;
    void seed(const Vector& joint_rad);
    void reset();
    void set_initial_hint(const Vector* hint);
    void configure_differentiation();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace motorhand::native::sharpa
