# 原生遥操驱动阻抗与自由空间摩擦补偿

`force-contact` 默认使用 `driver-impedance`。自由空间发送零刚度、零阻尼及运动摩擦前馈；检测到受阻后，由电机内部 MIT PD 维持局部代理目标，接触轴不叠加摩擦补偿。原生单输出线程负责速度滤波、代理状态与串口提交，不改变仿真源码或现场 FFB 零位。

```bash
# 在项目根目录启动；此命令会实际回零并使能两侧，请在完成现场检查后自行执行。
MOTORHAND_GLOVE_DRIVER=2 pixi run wuji-force-contact \
  --hand right --wuji-serial WH2KA01260813020 --approved-firmware 2.13.1 \
  --rate 50 --curves-port 8765 --feedback-mode driver-impedance --feedback-hardness 0.3

# 保留原来的接触估计→投影→MIT前馈路径。
MOTORHAND_GLOVE_DRIVER=2 pixi run wuji-force-contact \
  --hand right --wuji-serial WH2KA01260813020 --approved-firmware 2.13.1 \
  --rate 50 --curves-port 8765 --feedback-mode torque
```

`force-linear` 继续使用原来的力矩路径。`--feedback-hardness` 必须在 `[0,1]`，默认 `0.3`；它是人工设定，**不是**从电流估计出的物体硬度。普通轴基础刚度为 `5 + 55 × hardness` Nm/rad，默认 `21.5`；仿真基础倍率J1为1.5，按用户拇指增强20%的要求，新模式J1乘 `1.8`、J2–J4乘 `1.2`，其余乘1.0（旧虚拟阻抗的20%调整不在新模式重复叠加）。配置位于 `wuji-driver-impedance-right.json`，逐轴参数按原始电机 ID 排列。

接触检测继续使用既有电流、跟踪误差、停滞及迟滞判据。逐轴可施加阻抗mask排除释放延时及主动回开状态；旧contact仍保留用于抑制释放中耦合轴的摩擦，避免误差过零后生成反向代理。代理由最终 Wuji 目标减去实测位置的受阻误差求得：使用经过零位对齐、关节限位和启动混合后的最终 Jacobian，构造 `A = J_final × M⁻¹`，仅选取受阻行，求正则化的局部电机偏移并限制到每轴 `0.15 rad`。代理以 `1 rad/s` 更新，仅有关轴进入阻抗，其他有效自由轴继续摩擦补偿。该代理是受阻位置的局部近似，不能当作实测接触面。

自由空间摩擦为库仑项 `Fc × tanh((|dq| - deadband) / scale)` 加可选黏性项，方向跟随运动。默认普通轴 `Fc=0.25 Nm`，J2为 `0.10 Nm`，速度死区 `0.02 rad/s`、平滑尺度 `0.08 rad/s`、速度低通 `12 Hz`、摩擦上限 `0.6 Nm`。速度差分使用每台电机真实新收包时间（G4直接读取SDK解析位置反馈时的steady_clock时间，不以快照轮询时间替代，也不冒充电机内部采样时间）；重复包不更新滤波，超期和模式切换会清空历史。不能由该前馈推断手指意图或补偿静摩擦。

驱动端按 `a=sign×scale` 转换：`q_raw=zero+q_model/a`、`kp_raw=a²kp_model`、`kd_raw=a²kd_model`、`tau_ff_raw=a×tau_ff_model`。目标角仍受现场软件限位约束。力矩限额从 `wuji-max-torque-linear-right.json` 读取：2026-11-13按用户要求，J1/J2各 `10 Nm`，J3–J12各 `4 Nm`。保留 `torque_margin=0.85`，对应提交名义预算约 `8.5/3.4 Nm`，另扣编码余量。

规划与提交两层均计算总名义输出，包含弹簧、阻尼、前馈和 MIT 编码余量。提交层解码实际量化的目标、刚度、阻尼、速度零码和前馈零码，超出名义预算时缩小增益，撤离时若当前或预测总输出可能向内拉则完全撤力。无效几何、缺帧、切换、故障及急停沿用原始 `100 ms` 时限和零电流卸力；零输出使用精确 `0 A` 伺服帧，避免将 MIT 零码的半LSB误差当作真实零力矩。

**MIT PD没有硬力矩上限。** 上述预算仅针对提交计算所用状态的名义估算；命令存续期间真实位置或速度继续变化，实际输出仍可能超过该估算。局部代理、滤波速度和名义PD都不能替代力传感器。此次验证全部离线，未启动硬件；手感、撤离响应及实际摩擦仍需现场验证。

遥测保留原有 `schema_version=1` 与旧字段，另增加：

| 字段 | 含义 |
|---|---|
| `feedback_control_mode` | `driver_impedance` 或 `torque` |
| `feedback_hardness` / `feedback_hardness_source` | 人工设置的硬度值 / `manual_setting` |
| `glove_target` | 新模式为量化后发送时总名义输出 Nm；旧模式仍为量化MIT前馈 |
| `glove_target_semantics` | 新模式为 `nominal_total_pd_at_submission` |
| `glove_projected_torque_nm` / `glove_requested_torque_nm` | 新模式均为规划器的名义总输出；不包含未使用的旧反射力矩；语义字段为 `planned_nominal_total_pd` |
| `glove_friction_torque_nm` | 实际提交的自由空间摩擦前馈；接触轴为零；不把零码残差称为摩擦 |
| `glove_driver_pd_torque_nm` | 实际编码参数配合提交准备时反馈位置及估算速度计算的PD部分，不是实测力矩 |
| `glove_driver_target_rad` | 原始电机绝对目标角，含现场zero，单位rad |
| `glove_driver_kp` / `glove_driver_kd` | 量化后原始电机轴刚度 / 阻尼 |
| `glove_driver_estimate_received_ns` | PD位置反馈使用的逐轴真实收包时间 |
| `glove_driver_velocity_received_ns` | 滤波速度/摩擦计算使用的逐轴真实收包时间 |
| `glove_current_received_ns` | 原始电流独立收包时间；未知为null，过期电流值为null但保留原时间；不冒充位置时间 |

以上新增逐轴数组均按原始电机 ID `1..12` 排列。UDP发送上限扩为32768字节，保留完整矩阵与数值精度，不进行降采样。`glove_output_source_ns` 仍是该已完成输出的原始最老输入时间；`glove_submit_ns` 是最后一轴的主机提交时刻：驱动1为串口写入，驱动2为SDK接收命令，后续周期才到CAN；均不是电机执行确认。`virtual_hand_torque_nm`、电流与投影矩阵继续保留为判据/诊断信息，驱动阻抗模式不把旧虚拟反射力矩叠加到输出。

离线验证命令：

```bash
pixi run test-control-native
```

测试覆盖纯规划器、真实收包时间滤波、受阻局部代理、自由摩擦、PTY编解码和坐标变换、总PD预算、撤离/软件限位、异步线程切换与故障、完整Session接入、CLI参数、UDP字段与大小，以及旧torque/linear/inference回归。PTY测试仅打开测试创建的伪终端。
