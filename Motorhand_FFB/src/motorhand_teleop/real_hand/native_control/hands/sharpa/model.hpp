#pragma once
#include "common/hand.hpp"
#include <filesystem>
#include <memory>

namespace motorhand::native::sharpa {
// SDK/URDF顺序22轴弧度；hand_points是校准后的手套任务域米，物理FK单独提供。
class Model final : public HandModel<22> {
public:
    Model(std::filesystem::path asset_root, bool left, std::filesystem::path config_path = {});
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const std::array<std::string,22>& joint_names() const;
    const JointArray<22>& lower_limits() const;
    const JointArray<22>& upper_limits() const;
    const JointArray<22>& effort_limits() const;
    JointArray<22> solve(const Points&) override;
    // 实测FK只要求有限角度，保留动力学软限位超调；命令范围由solve保证。
    Points hand_points(const JointArray<22>&) override;
    Points physical_hand_points(const JointArray<22>&);
    // 行优先63x22，包含重定向时固定的轴，供实际反馈几何与离线验证使用。
    std::array<double,63*22> point_jacobian(const JointArray<22>&);
    std::array<double,21> task_scales() const override;
    // 接受有限实测角，内部先验投影到命令范围，不改变实测输入。
    void seed(const JointArray<22>&) override;
    void reset() override;
    std::unique_ptr<HandModel<22>> create_solver(bool differentiation=false) const override;
    void set_initial_hint(const JointArray<22>*) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace motorhand::native::sharpa
