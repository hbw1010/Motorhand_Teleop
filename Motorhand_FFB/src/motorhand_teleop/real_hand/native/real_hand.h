#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI 6。仅 model=2 (Wuji Hand2)；hand: 0=left，1=right。
 * 构造不连接设备。字符串由 create 复制，调用者保留原有所有权。
 * 关节顺序始终为 thumb/index/middle/ring/pinky，各 S1..S4。
 */
typedef struct RealHandOptions {
    int model;
    int hand;
    const char *serial_number;
    const char *address;
    const char *approved_firmware_version;
    /* 同设备官方 SDK 只读预检读到的版本；不是批准值的副本。
     * C SDK 尚无该读取接口，调用者负责先完成预检。enable要求与批准值一致。
     */
    const char *preflight_firmware_version;
    /* 可选项目回零轨迹约束；两者同时为0表示不另加速度/加速度限制。
     * 回零仍使用五次渐变参考；不是官方默认速度或实体安全保证。显式配置时两者须为正。
     */
    double max_velocity;
    double max_acceleration;
    /* 可选实时主机速度上限rad/s；0表示不附加主机限速，不添加项目滤波。 */
    double realtime_max_velocity;
    double command_timeout;
    double feedback_timeout;
    double home_rate_hz;
    double home_tolerance;
    double home_timeout;
    double enable_timeout;
    /* 回零固定起点五次渐变时长（秒，正有限数）；Python默认1.5，不影响正常send。 */
    double transition_duration;
    /* -1保留设备现有配置；MIT kp/kd必须同时为-1或同时显式设置。 */
    double effort_limit;
    double mit_kp;
    double mit_kd;
    uint32_t timeout_ms;
    /* 0=调用者显式管理home；1=独立原生会话enable/正常close自动home。 */
    uint8_t auto_home;
} RealHandOptions;

/* phase: 0=disconnected, 1=connected, 2=enabled, 3=fault, 4=closed。
 * positions 始终为实测值；速度rad/s与effort电流A来自同一joint_states帧。
 * has_velocity/has_effort仅在对应20轴全部有限且可用时为1，否则数组不可使用。
 * timestamp_ns 为该帧主机单调接收时间，不能读时刷新；device_timestamp_us
 * 为同帧设备时间，0 表示设备接口未提供。effort不是独立接触力或力矩传感值。
 * firmware_version 为同一设备只读预检提供的实测版本，不声称由C接口读取。
 */
typedef struct RealHandState {
    double positions[20];
    double velocities_rad_s[20];
    double effort_a[20];
    uint8_t has_velocity;
    uint8_t has_effort;
    double lower_limits[20];
    double upper_limits[20];
    uint8_t enabled[20];
    uint8_t online[20];
    uint32_t error_codes[20];
    uint64_t timestamp_ns;
    uint64_t device_timestamp_us;
    int phase;
    char firmware_version[128];
} RealHandState;

typedef struct RealHand RealHand;

/* 所有错误通过返回值及调用方缓冲区报告，不允许异常越过 C ABI。
 * 0=成功，-1=非法参数，-2=运行/设备错误；create 失败返回 NULL。
 */
int real_hand_abi_version(void);
RealHand *real_hand_create(const RealHandOptions *options, char *error, size_t error_size);
int real_hand_connect(RealHand *hand, char *error, size_t error_size);
/* 显式接管：核准固件，失能并确认，清错一次并核验，再使能。
 * 仅auto_home=1随后回零。不修改硬件原点；停止级/未知故障不得进入运动。
 * 回零必须零命令后的新鲜实测位置到位；在线、帧龄和使能检查不放宽。
 */
int real_hand_enable(RealHand *hand, char *error, size_t error_size);
/* 默认直传官方位置发布器；显式realtime_max_velocity>0才附加单步限速。
 * 不施加项目死区、滤波或跨帧加速度约束。
 */
/* 在原生控制锁内及实际发送前核验monotonic期限，
 * 期限到后直接停机锁存，不把源失效降级成正常回零。0不是有效期限。
 */
int real_hand_send(RealHand *hand, const double *positions, size_t count,
                          uint64_t deadline_ns, char *error, size_t error_size);
int real_hand_read_state(RealHand *hand, RealHandState *state, char *error, size_t error_size);
/* 回零按home_rate_hz发布transition_duration五次渐变参考，之后持续保持零目标。
 * 必须发布最终零命令后再以新鲜实测位置到位确认；home_timeout覆盖轨迹和等待。
 * 默认路径不另加速度/加速度限速；未到位超时报告容差及各关节实测偏差。
 */
int real_hand_home(RealHand *hand, char *error, size_t error_size);
/* disable/emergency_stop 是直接停止路径，不回零；急停后独立请求失能并核验。
 * 未经本实例显式enable授权的只读会话，停止操作不下发任何设备命令。
 * 故障收尾已经确认失能并释放连接时，重复停止幂等成功，不清除原控制故障。
 * 未确认失能仍返回失败；幂等成功不是对其他控制器后续操作的状态承诺。
 */
int real_hand_disable(RealHand *hand, char *error, size_t error_size);
int real_hand_emergency_stop(RealHand *hand, char *error, size_t error_size);
/* 正常close仅auto_home=1先回零，再禁用/释放；故障绝不回零。
 * 从未enable的只读会话只释放，不禁用其他控制器已使能的设备。
 * Python默认auto_home=0，enable/close均不隐式回零。
 * 已交付的锁存故障不重复报告；未交付的看门狗故障在此报告一次。
 */
int real_hand_close(RealHand *hand, char *error, size_t error_size);
/* 释放内存；未显式 close 时仅尽力急停/释放，析构不隐式产生回零运动。 */
void real_hand_destroy(RealHand *hand);

#ifdef __cplusplus
}
#endif
