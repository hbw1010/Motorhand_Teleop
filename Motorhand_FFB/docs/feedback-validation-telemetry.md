# 六组验证曲线的原生遥测契约

本次仅增加观测数据；阻抗参数、限幅数值、控制运算与编码命令保持原样。沿用 schema_version=1，增加的字段由已有 completed 输出快照、反馈请求 ID 与真实计算分支提供，不以界面发布时的最新反馈替代规划输入。

## 坐标与时间

新增手套数组均为原始电机 J1–J12 顺序（12 项），Wuji 数组沿用 `wuji_joint_names` 顺序（20 项）。内部模型电机角转换为原始绝对电机角：`q_raw = zero_deg * π / 180 + q_model / a`，其中 `a = sign * scale`；速度除以 `a`，阻挡方向乘以 `sign(a)`。原始目标及代理角包含现场 zero，不能与模型指节角混用。已有增益转换仍为 `kp_raw=kp_model*a²`，力矩仍为 `tau_raw=tau_model*a`。

所有时刻均为主机单调时钟 ns。真实收包时间不是设备内部时间；软件接触判据完成时刻不是物理接触时刻；SDK 提交完成时刻不是执行完成时刻。

| 字段 | 来源和意义 |
| --- | --- |
| `feedback_request_id` | 产生这次 completed 输出的请求；以下规划数据均属于该请求 |
| `wuji_feedback_received_ns` | 请求携带的 Wuji 原始反馈收包时刻 |
| `contact_evaluated_ns` | `ForceController::step` 返回后记录的主机时刻；该调用即使返回无效几何也记录 |
| `impedance_planned_ns` | `DriverImpedancePlanner::step` 返回后记录的主机时刻，包括已完成的安全撤力规划 |
| `glove_proxy_source_ns[12]` | 规划请求实际使用的每轴位置收包时刻 |
| `glove_driver_velocity_received_ns[12]` | 现有字段；规划器本次更新速度所使用的位置时刻。规划提前释放且未更新速度时为 null |
| `glove_driver_estimate_received_ns[12]` | 输出线程估算最终编码名义力矩所用的位置时刻，可能晚于规划输入 |
| `glove_submit_ns` | 12 轴实际 SDK 提交中的最后完成时刻，维持现有语义 |
| `glove_current_received_ns[12]` | 独立电流收包时刻，不能拿位置时间替代 |

## 代理、方向和状态

| 字段 | 语义 |
| --- | --- |
| `glove_proxy_surface_rad[12]` | 规划器实际保留的阻挡参考表面，原始绝对电机 rad；不是最终下发目标 |
| `glove_proxy_input_rad[12]` | 本次规划实际输入位置，原始绝对电机 rad；不随最新 UI 帧变化 |
| `glove_filtered_velocity_rad_s[12]` | 同次规划速度估计器的输出，原始电机方向 rad/s；不能用最终撤力命令中的零速度冒充 |
| `glove_blocking_direction[12]` | 原始电机按入方向，取 -1、0、1；沿阻挡方向越过参考的位移为 `direction*(q_raw-proxy_raw)` |
| `glove_proxy_valid[12]` | 本轴存在有效受阻参考；自由轴、无几何和整次规划失败均 false。保留参考但输出保护撤力时仍可 true |
| `impedance_contact[20]` | 实际加入本次局部逆解 normal/rhs 的 Wuji 行；方向不符的候选行不算，整次规划失败时清空 |
| `contact_active_mask[20]` | 现有接触锁存状态，可在释放迟滞中继续 true；不代表该行继续参与阻抗求解 |
| `glove_torque_limited[12]` | 实际命中弹簧/阻尼总预算、自由摩擦上限、输出端增益上限、前馈或量化后预算缩放 |
| `glove_motion_limited[12]` | 规划/提交目标命中位置范围限制，或因当前位置处于软限位且名义力矩朝外而撤力 |
| `glove_proxy_limited[12]` | 代理偏移或表面移动速率受限；位置软限位单独记到 motion_limited |
| `glove_control_reason[12]` | 实际执行分支的原因字符串，可用 `\|` 连接；规划与提交阶段原因合并 |

每个原因均在该分支执行时记录，不由波形、当前角差或上游接触状态推测。`contact_pd` 只表示规划时形成了 PD；后续可能同时出现输出撤力原因。

| 原因 | 对应分支 |
| --- | --- |
| `inactive_geometry` | 上游 `active` 为 false，含没有可用 Jacobian 的请求；不细分上游为何无效 |
| `invalid_input` | 规划输入位置非有限，或位置源时间无效/超期 |
| `invalid_geometry` | 输入矩阵或变换后的矩阵非有限 |
| `unobservable_geometry` | 最终 Jacobian 总范数低于原控制阈值 |
| `singular_transmission` | 电机传动矩阵不可逆或条件不足 |
| `invalid_contact_row` | 被提名的受阻行输入非有限或该行局部映射不可观测，整次规划释放 |
| `inverse_failed` | 局部逆解结果非有限 |
| `contact_pd` | 规划器本轴返回非零阻抗增益 |
| `free_axis` | 本轴无受阻方向，且不与锁存接触行耦合 |
| `coupled_release` | 本轴无受阻逆解方向，但仍与锁存接触行耦合，因此不补摩擦；也涵盖局部逆相互抵消，不推测人手意图 |
| `friction_disabled` | 自由轴的摩擦补偿配置为关闭 |
| `outside_proxy` | 有阻挡方向，但沿该方向未超过代理表面 |
| `zero_stiffness` | 本次有效刚度为零 |
| `planner_release_guard` | 代理作用距离、当前总输出或预测输出命中原有完整撤力保护；不宣称一定是主动回开 |
| `proxy_offset_limited` / `proxy_rate_limited` | 代理偏移/表面移动速率实际被截断 |
| `soft_limit_target` / `soft_limit_release` | 目标位置限制/位置边界处朝外输出导致撤力 |
| `torque_budget_limited` / `friction_limited` | 总预算或自由摩擦幅值实际截断 |
| `sdk_gain_limited` | 输出端原始 kp/kd 命中 MIT 范围限制 |
| `sdk_zero_output` | 输出端作用项为零或前馈小于半编码步，走精确零电流分支 |
| `sdk_direction_release` | 量化后当前或预测名义输出方向触发保护，改发零电流 |
| `sdk_budget_release` | 量化预算缩放尝试耗尽后改发零电流 |
| `sdk_output_accepted` | 输出端接受本轴量化后的 MIT 阻抗命令 |
| `unknown` | 没有对应 completed 观测或没有记录到原因，不制造轴级失败解释 |

## 缺失、失败与边界

无阻抗 completed 输出时，12 项代理角、规划输入、源时间和滤波速度均为 null，参考有效及三个限幅标志为 false，方向为 0，原因为 unknown；`impedance_contact` 为 null，表示未知。存在完成的释放规划时，`impedance_contact` 是 20 项 false，参考角为 null，但保留该请求的输入、源时间与真实释放原因。不能把“未知”与“已求解且无参与行”合并。

worker 异常或未完成提交时不制造 completed 轴级原因；终止事件使用现有 `teleop_status`。电流继续遵守原有独立源时间及 100 ms 有效窗。

遥测仍限制为完整单包 32 KiB，不裁剪数组、不降低浮点精度。离线契约测试包含完整 240 项矩阵、19 位时刻、长科学计数数值和全部原因字符串；测试直接经 localhost UDP 验证到达与包长。

这些字段支持代理位置、名义输出、软件时序和保持稳定性的核验。两侧电流比例不是验收标准；实测电流转力矩仍需独立标定，默认的名义模型或空载参考不能充当独立执行准确性证据。
