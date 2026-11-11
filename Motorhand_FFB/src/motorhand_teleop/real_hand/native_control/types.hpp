#pragma once

#include <array>
#include <cstdint>

namespace motorhand::native {

using Vec12 = std::array<double, 12>;
using Vec20 = std::array<double, 20>;
using Vec68 = std::array<double, 68>;
using Points = std::array<double, 63>;
using PointJacobian = std::array<double, 63 * 12>;
using MotorJacobian = std::array<double, 12 * 12>;

enum class ContactPhase { not_configured, free, arming, contact, releasing, waiting_geometry };

// 数组顺序分别固定为项目 GLOVE_JOINTS、WUJI_JOINTS 与 CAD 模型树顺序。
// 电机位置为模型轴相对角 rad；原始编码器位置为 degree。时间为单调时钟真实收包 ns。
struct GloveFrame {
    Vec12 raw_degrees{};
    Vec12 motor_rad{};
    Vec12 measured_current_a{};
    std::array<std::uint64_t, 12> received_ns{};
    std::uint64_t oldest_ns{};
};

// points 是腕部平移后的 canonical 21 点，单位 m；雅可比按行主序。
struct GlovePose {
    Vec12 joint_rad{};
    Vec12 motor_rad{};
    Vec68 tree_rad{};
    Points points_m{};
    PointJacobian point_jacobian{};
    MotorJacobian motor_jacobian{};
    double max_gap_m{};
};

struct HandFrame {
    Vec20 position_rad{};
    Vec20 velocity_rad_s{};
    Vec20 effort_a{};
    bool has_velocity{};
    bool has_effort{};
    std::uint64_t received_ns{};
};

}  // namespace motorhand::native
