# Wuji Hand 2 官方资料研究报告：力反馈/接管可行性与约束

> **研究目的**：为自研屈伸电机手套驱动 Wuji Hand 2 的力反馈/接管方案建立一手资料基线。本文只把舞肌科技（WUJI TECH）官网、官方 Docs Center、`wuji-technology` 官方 GitHub 仓库中的内容作为“官方明确事实”；对接线、映射和安全架构的建议单独标记为“基于资料的推断”。
>
> **资料访问日期**：2026-11-01。官方文档当前同时存在 Wuji Hand 2.1 (Beta) 与 2.2 (Beta) 版本；以下涉及接口与安全的主体内容以 **2.2 (Beta) 最新文档**为准，除非特别注明。

## 1. 结论摘要

1. **真机结构**：官方把 Wuji Hand 2 定义为 20 个主动自由度、五指、每指 4 个独立主动关节；拇指支持对掌，四指支持侧摆。关节是可反驱的直驱旋转关节。来源：[产品介绍](https://docs.wuji.tech/docs/en/wuji-hand/latest/overview.md)、[官网产品页](https://www.wuji.tech/en/hand2)（访问：2026-11-01）。
2. **官方可直接使用的控制面**：官方 Wuji SDK 提供 Python 与 C 接口。整手 `joint_command` 每次发送恰好 20 项，每项含 `position`（rad）、`velocity`（rad/s）、`effort`（A，电流代理/前馈）；状态从 `joint_states` 读取位置、速度和 `effort`，而 `joint_diagnostics` 提供电流、电压、温度、状态字、故障码和通信统计。来源：[SDK 数据流总览](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference.md)、[关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md)、[诊断](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/diagnostics.md)（访问：2026-11-01）。
3. **控制器性质**：设备默认运行 MIT 力位混合控制，官方给出的关系式为 `τ = Kp·(q_d−q) + Kd·(dq_d−dq) + τ_ff`。可通过 SDK 读写每关节 `kp/kd` 和 `effort_limit`；`effort` 的官方单位是 A，不应未经标定当作 N·m 或接触力。官方明确警告 2.2 原型的电流到外部关节力矩/接触力闭环性能尚未验证。来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md)、[力矩与 MIT 参数](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/effort-and-mit.md)、[用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md)（访问：2026-11-01）。
4. **通信和接管边界**：2.2 通过 RJ45、100BASE-TX 以太网通信；默认静态地址为左手 `192.168.1.110`、右手 `192.168.1.111`，默认数据端口 `50001`，不支持 DHCP。多客户端共享默认为 `enable_bridge=True`，但自研接管程序仍应按序列号明确选机并建立独占策略。来源：[电气与网络集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/electrical-network-integration.md)、[SDK 设备连接](https://docs.wuji.tech/docs/en/wuji-sdk/latest/device-connection.md)（访问：2026-11-01）。
5. **失联/急停不是“自动恢复”**：使能期间通信丢失会停止电机并报故障，恢复网络后不会自动继续，必须清理原因并再次 `enable()`；软件 `emergency_stop()` 是整手立即停止且不带 mask；当前版本没有硬件急停接口。官方 ROS 2 驱动另外明确声明没有 command watchdog，停止发布者不会自动发零、急停或禁用，不能把进程退出当作安全停机。来源：[维护](https://docs.wuji.tech/docs/en/wuji-hand/latest/maintenance.md)、[设备操作](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/device-operations.md)、[ROS 2 集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/ros2-integration.md)（访问：2026-11-01）。
6. **力反馈结论**：官方资料支持“用 MIT 参数、位置/速度目标和电流前馈做设备原生混合控制”，但**没有公开可直接依赖的外部力传感器/关节力矩闭环保证**；2.2 非触觉样机的电流不能据此换算接触力。若“力反馈”指人手套感知力后接管手部动作，仍需用户自己的传感器闭环和真机标定；若指用手套直接生成关节目标，则可复用官方重定向/关节命令接口，但不得假设官方提供了手套到电机的私有协议。来源：[用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md)、[快速开始](https://docs.wuji.tech/docs/en/wuji-hand/latest/quick-start.md)（访问：2026-11-01）。

## 2. 版本和资料适用范围（必须先确认）

| 项目 | 官方明确事实 | 对本项目的含义 |
|---|---|---|
| 最新手册 | 最新 `wuji-hand/latest` 手册覆盖 **Wuji Hand 2.2 (Beta) 非触觉原型**；参数和表现不代表最终产品。2.1 设备应看 `beta1` 手册。 | 首次连接必须记录序列号、固件版本和硬件版本，不能只按“Hand 2”名称套参数。 |
| 序列号识别 | 序列号第 5–7 位为 `A00` 表示 2.1，`A01` 表示 2.2。 | 资产、关节模型、固件和 SDK 兼容性选择应绑定实际序列号。 |
| 2.2 软件/硬件 | 2.2 固件从 v2.0.0 起（页面当前列 v2.5.1）；2.1 固定 v1.2.4；官方建议固件和 SDK 一起升级。 | 任何速率控制、端口写入或诊断字段都要以实机版本矩阵复核。 |
| 2.2 接口变化 | 2.2 当前为 XT30 + RJ45 两线布局；官方警告后续 2.2 版本会改变外部接口形式和供电范围。 | 机械/电气设计不要把当前 XT30/RJ45 视作永久产品合同。 |

来源：[版本识别与兼容性](https://docs.wuji.tech/docs/en/wuji-hand/latest/version-compatibility.md)、[产品介绍](https://docs.wuji.tech/docs/en/wuji-hand/latest/overview.md)（访问：2026-11-01）。

> **版本冲突提示**：官网产品页仍将脚注规格称为“当前 Beta1”，而最新产品介绍针对 2.2 Beta 非触觉样机并明确“非最终产品”。官网页给出 `≥12 N` 指尖峰值力，不能直接当作当前 2.2 真机承诺；用户须知更明确地说原型电机峰值/额定力和负载曲线尚不构成外部设计或采购依据。来源：[官网产品页](https://www.wuji.tech/en/hand2)、[用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md)（访问：2026-11-01）。

## 3. 真机自由度、关节命名和运动范围

### 3.1 自由度事实

- 20 个主动 DoF、20 个关节，5 个手指各 4 个；每个主动关节独立驱动，拇指支持对掌（opposition），四指支持 MCP 侧向外展/内收。运动学为串联直驱旋转关节，无非线性机械耦合。来源：[产品介绍](https://docs.wuji.tech/docs/en/wuji-hand/latest/overview.md)（访问：2026-11-01）。
- 三类官方名称同时存在：SDK 使用 `{finger}_S1..S4`；模型使用解剖名称；模型执行器使用 `{l,r}_{THJ|FFJ|MFJ|RFJ|LFJ}{0-3}`。官方明确提示：**SDK 的 S1..S4 与模型执行器 J0..J3 的轴序映射尚未发布**，不能凭名称或顺序猜测。来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#joint-table)（访问：2026-11-01）。

### 3.2 官方 SDK 扁平索引（0–19）

| 扁平索引 | SDK 标签 | 关节功能（S1 → S4） | 模型执行器前缀 |
|---:|---|---|---|
| 0–3 | `thumb_S1..S4` | `cmc_flex`、`cmc_abd`、`mcp`、`ip` | `{l,r}_THJ0..3` |
| 4–7 | `index_S1..S4` | `mcp_flex`、`mcp_abd`、`pip`、`dip` | `{l,r}_FFJ0..3` |
| 8–11 | `middle_S1..S4` | 同四指 | `{l,r}_MFJ0..3` |
| 12–15 | `ring_S1..S4` | 同四指 | `{l,r}_RFJ0..3` |
| 16–19 | `pinky_S1..S4` | 同四指 | `{l,r}_LFJ0..3` |

来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#joint-table)、[关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md#joint-numbering)（访问：2026-11-01）。

### 3.3 官方关节范围（模型和左右手镜像）

| 关节组 | 范围 |
|---|---:|
| 拇指 `cmc_flex` | −68° ～ 74° |
| 拇指 `cmc_abd` | −85° ～ 40° |
| 拇指 `mcp` | −60° ～ 90° |
| 拇指 `ip` | −60° ～ 90° |
| 四指 `mcp_flex` | −60° ～ 90° |
| 四指 `mcp_abd` | −40° ～ 40° |
| 四指 `pip` | −60° ～ 120° |
| 四指 `dip` | −60° ～ 90° |

官方称这些范围与 URDF/MJCF/USD 模型 joint limits 一致，且左右手镜像。来源：[产品介绍](https://docs.wuji.tech/docs/en/wuji-hand/latest/overview.md#joint-motion-direction-and-range)、[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#joint-table)（访问：2026-11-01）。

### 3.4 真机映射不能只用在线数组位置

`joint_states` 和 `joint_diagnostics` 都是**可变长度、只含在线关节**的帧；每项通过 `nid` 识别。C API 特别说明 `nid` 是固件流编号，不是 0–19 扁平索引，并提供 `wuji_hand_2_nid_to_joint_index()` / `wuji_hand_2_joint_index_to_nid()` 转换。因而接管程序必须按 `nid` 建立状态映射，并处理少于 20 个在线关节的情况。来源：[关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md)、[诊断](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/diagnostics.md)（访问：2026-11-01）。

## 4. 官方命令与状态接口（不猜 SDK）

### 4.1 连接、发现和设备选择

官方 SDK 公开 Python 与 C 两套接口；官方 GitHub README 说明 Python 是完整主接口，C 是共享库（Linux `libwuji_sdk_c.so`、macOS `libwuji_sdk_c.dylib` 和 `wuji_sdk.h`），示例分别位于 `examples/python/wuji_hand_2/`、`examples/c/wuji_hand_2/`。来源：[官方 wuji-sdk 仓库](https://github.com/wuji-technology/wuji-sdk)（访问：2026-11-01）。

官方推荐先扫描，再按 `sn`、地址或左右手选择设备；同一手别有多个设备时，按 handedness 连接会报 `AmbiguousHandedness`，应改按序列号。连接选项示例为 `timeout_ms=1000`、`retry_count=3`。`device_name` 只是本地别名，不决定设备类型。来源：[SDK 设备连接](https://docs.wuji.tech/docs/en/wuji-sdk/latest/device-connection.md)（访问：2026-11-01）。

### 4.2 状态流

| 官方资源 | 频率/数据 | 实施要点 |
|---|---|---|
| `hand.joint_states().subscribe()` | 原生 1000 Hz；每项 `nid`, `position` (rad), `velocity` (rad/s), `effort` (A) | 关节角和速度只应从此流读取；帧只含在线关节。 |
| `hand.joint_diagnostics().subscribe()` | 与 `joint_states` 同一设备流；每项状态字、电流 A、母线电压 V、MCU 温度 °C、当前故障码、关节总线响应率/超时数 | 不含关节角；必须另订阅 `joint_states`。 |
| `hand.imu().subscribe()` | 原生 100 Hz | 官方 SDK Hand 2 数据流。 |
| `hand.comm_diag().get()` | 1 Hz；5 指通信吞吐/错误率 | 可用于通信健康监控。 |
| `online_joints_count().get()` | 0–20 | 首次使能前应确认是否为 20。 |

订阅句柄可降低推送频率，`0` 恢复原生频率；官方要求固件与 SDK 版本同时满足速率控制条件。官方控制指南建议 GET 查询不超过 100 Hz，PUB 命令最高 1 kHz。来源：[数据流总览](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference.md)、[关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md)、[诊断](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/diagnostics.md)（访问：2026-11-01）。

### 4.3 关节命令

官方定义 `JointCommand`：

```python
class JointCommand:
    position: float  # rad，目标位置
    velocity: float  # rad/s，目标速度
    effort: float    # A，力矩/电流前馈字段
```

`hand.joint_command().publish().send(...)` 每次必须发送**恰好 20 项**，顺序为扁平索引 0–19；不需要前馈的字段填 0。C API 的对应入口为 `wuji_hand_2_joint_command_publish()`、`wuji_joint_command_publisher_send()`。官方 C 文档建议固定频率（通常 200 Hz–1 kHz）连续调用，并在结束时关闭 publisher 释放资源。来源：[关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md#joint_command-pub)（访问：2026-11-01）。

### 4.4 使能、禁用、原点和急停

官方公开整手或 20 位 mask 控制：

- `enable()` / `disable()`：整手使能/禁用；也可用长度 20 的 mask 只作用于选中关节。
- `clear_fault()`：清除整手或选中关节故障。
- `set_origin()` / `clear_origin()`：设置/清除用户原点；运动中的关节会延迟到下一个 IDLE 生效。
- `emergency_stop()`：整手立即停止，不接受 mask。

来源：[设备操作](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/device-operations.md)、[力矩与 MIT 参数](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/effort-and-mit.md)（访问：2026-11-01）。

## 5. 力位混合和“力反馈”边界

### 5.1 官方明确支持的控制

- 设备默认 MIT force-position hybrid control，Python API 不切换控制模式。
- 官方公式：`τ = Kp · (q_d − q) + Kd · (dq_d − dq) + τ_ff`。
- `q_d`、`dq_d`、`τ_ff` 分别来自 `joint_command` 的 `position`、`velocity`、`effort`。
- `kp`、`kd` 可通过 `mit_params` 按统一值或 20 关节数组读写；`effort_limit` 可按统一值或每关节数组读写。
- 写入拒绝 NaN、Inf 和负值；`effort_limit` 还不能超过固件设备上限。最新控制指南明确给出 `effort_limit` 硬上限为 **2.0 A**。
- 电源重启会恢复 `kp`、`kd`、`effort_limit` 默认值；SDK 写入只作用于当前会话，连接后 SDK 会读回 `kp/kd` 并写入日志。

来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#mit-control-mode)、[力矩与 MIT 参数](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/effort-and-mit.md)（访问：2026-11-01）。

### 5.2 官方明确不支持/不承诺的内容

- `joint_states.effort` 与 `joint_diagnostics.current` 的单位都是 A（电流），不是官方声明的 N·m 测量值。
- 2.2 用户须知明确写出：用关节电流估计外部关节力矩的正向路径虽然存在，但硬件一致性与标定精度尚未收敛，闭环表现尚未验证；**不要用关节电流判断外部接触力或力矩**。
- 用户须知同时声明热设计、长期满载/高载运行、可靠性、耐久性、峰值/额定力和负载曲线仍在迭代或未验证。
- 当前维护页说明无硬件急停接口；软件急停后控制应用仍应停止继续发命令。

来源：[用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md#capability-boundaries-and-items-not-yet-provided)、[维护](https://docs.wuji.tech/docs/en/wuji-hand/latest/maintenance.md#emergency-stop)（访问：2026-11-01）。

### 5.3 最大负载占比近似的参数依据

- [官方控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md)再次核实：
  effort单位A，effort_limit设备上限2.0A。2.0A可作电流归一化参考，不是最大Nm。
- [官方用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md)未给出可用的每关节峰值Nm/Kt表，
  且明确不应把2.2原型电流当作外部接触力判断依据；本项目近似模式不声称绕过这项限制。
- 本地Encos `hardware/encos_dexhand/src/encos_hand/protocol.py` 的±10Nm只决定编码范围；
  未发现该SDK提供实际电机型号、额定/峰值Nm或Kt。12位编码的步长20/4095Nm不能忽略。
- 因此采用“电流占比＋几何投影→独立软件允许的手套名义Nm”，不生成假Wuji Nm或 `calibrated=true`。
  初始0.01Nm是软件试验限额，不是从URDF或协议上限推断出的电机能力。

来源核对日期：2026-11-03。只查静态资料，未连接真实设备。

### 5.4 对“手套驱动/接管”的可执行解释（推断）

> **以下是基于上述官方接口的工程推断，不是官方承诺。**

- 若“接管”是手套输出人手姿态：可在自研程序中生成 20 个 `JointCommand.position`，按官方 0–19 顺序持续发布；SDK 官方还提供 21 个手部关键点到 20 个 Hand 2 关节角的 retargeting 示例，但自研手套的坐标系、手性、零点、限位和动态滤波仍需独立标定。来源：[官方 SDK 发布记录](https://github.com/wuji-technology/wuji-sdk/releases)、[快速开始](https://docs.wuji.tech/docs/en/wuji-hand/latest/quick-start.md#cross-product-teleoperation)（访问：2026-11-01）。
- 若“力反馈”是根据手套/外部传感器施加阻抗：可将期望位置、速度和 `effort` 前馈送入 MIT 控制接口，但官方没有公开“电机电流→外力”的可信标定合同。因此应把外部力传感器或手套力传感器作为独立测量通道，不把 Hand 2 的 `effort` 当作闭环接触力。
- 若需在某些关节失效时继续驱动在线关节：官方接口允许在线计数、可变长度状态帧和按 mask 使能/禁用，但每次 `joint_command` 仍需 20 项；应用层必须把离线关节安全填值并禁止对故障关节持续写入。这是安全架构推断，须真机确认故障时固件实际接受哪些目标。

## 6. 通信、供电和多客户端约束

### 6.1 物理与网络参数

| 项目 | 2.2 官方值/行为 |
|---|---|
| 供电 | 11–13 V DC，12 V nominal；随附 12 V 20 A 适配器，额定至少 200 W |
| 电源接口 | XT30；官方随附 14 AWG、2 m 公母延长线 |
| 通信 | RJ45，100BASE-TX Ethernet |
| 默认 IP | 左手 `192.168.1.110`；右手 `192.168.1.111` |
| 子网/网关 | 子网掩码 `255.255.255.0`；网关 `192.168.1.1` |
| 默认数据端口 | `50001`；设备接受 SDK 数据连接的 UDP 端口 |
| 地址分配 | 静态 IP，不支持 DHCP |

官方明确要求不要在运行时插拔电源/通信线、不要反接；XT30 极性图、工作/峰值/待机功耗、浪涌/保险/接地建议尚未提供。数据端口可从 10000–65535 设置，但排除发现端口 50000；改 IP/端口都要写入后重启才能生效。来源：[电气与网络集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/electrical-network-integration.md)、[用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md)、[设备操作](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/device-operations.md)（访问：2026-11-01）。

### 6.2 连接共享和端口安全

- SDK `enable_bridge=True` 默认允许多个 SDK 实例（包括其他主机进程）共享已连接设备；关闭后可使用独占模式。
- 同一手别多个设备必须按 `sn` 选择；自研接管程序不得用“扫描到的第一个设备”作为生产选择逻辑。
- 官方连接默认示例为 1 s 超时、3 次重试；这不是实时控制安全阈值，应由应用层另设命令周期、状态新鲜度和失联停机策略（后半句为推断）。

来源：[SDK 设备连接](https://docs.wuji.tech/docs/en/wuji-sdk/latest/device-connection.md)、[电气与网络集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/electrical-network-integration.md)（访问：2026-11-01）。

### 6.3 失联、故障和恢复

官方规定的控制闭环为“发送命令 → 读取状态 → 检查故障码 → 故障时停止”：

1. 通过 `joint_diagnostics` 读取错误码，使用 `WujiHand2.describe_error(code)` 获取 `severity` 和 `clear_policy`，不要自行按十六进制位猜严重级别。
2. `Warning` 不停机；`DeferredStop` 在当前动作结束后停；`ImmediateStop` 立即停；`Fatal` 停止且不可清除，需要断电。
3. `ManualClear` 要先排除原因再 `clear_fault()`；`AutoClear` 等待自动清除；`NonClearable` 需要断电，复现则停止使用并联系官方。
4. 通信丢失时设备停电机并报故障；网络恢复后不会自动恢复动作，需确认故障原因、清除并重新 `enable()`。

来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#safe-control-loop)、[维护](https://docs.wuji.tech/docs/en/wuji-hand/latest/maintenance.md#fault-severity-and-stop-behavior)、[电气与网络集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/electrical-network-integration.md#timeouts-disconnection-and-reconnection)（访问：2026-11-01）。

## 7. 官方模型、模型仓库和可引用资产

### 7.1 官方模型来源

官方 `wuji-technology/wuji-description` 仓库是 Wuji 机器人模型描述包，提供 URDF、MuJoCo MJCF/MJX、USD 和 STEP/CAD。仓库树包含 `hand2/hand2_1/` 与 `hand2/hand2_2/`，每个版本有左右手、MJCF、URDF、USD、网格和安装接口。来源：[官方 wuji-description 仓库](https://github.com/wuji-technology/wuji-description)（访问：2026-11-01）。

2.2 模型官方定义如下：

- ROS 2 包名：`wuji_hand2_2_description`，根目录 `hand2/hand2_2/body/`。
- 根链接：`l_mount` / `r_mount`，即机械臂法兰安装点；腕壳通过 `{l,r}_wrist_fixed` 固定到根链接。
- 模型名：`wujihand2_2-{side}`；USD 入口 `wujihand2_2.usd`。
- URDF、MJCF、USD 均含软垫几何的刚体模型；**不支持可变形体仿真**。
- 2.2 模型含 20 个旋转关节，可在 MuJoCo、ROS 2/RViz、Isaac Sim 使用。

来源：[模型资产与加载](https://docs.wuji.tech/docs/en/wuji-hand/latest/model-integration.md)、[官方 wuji-description 仓库](https://github.com/wuji-technology/wuji-description)（访问：2026-11-01）。

### 7.2 模型用于映射的限制

官方模型可作为姿态/限位/碰撞几何参考，但不能代替真机轴序、零点和符号验证：

- 官方尚未发布 SDK `S1..S4` 与模型 `J0..J3` 的轴序映射；必须让真机状态变化逐关节验证，并记录左右手符号。
- 2.2 官方模型是刚体仿真，软垫几何不等于已验证的接触力模型。
- 模型加载成功只能证明资产/可视化链路工作；官方明确说固件、网络、传感器和真机控制仍需连接设备验证。

来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#joint-table)、[模型资产与加载](https://docs.wuji.tech/docs/en/wuji-hand/latest/model-integration.md#verify)（访问：2026-11-01）。

## 8. 面向自研手套接管的约束清单（可落地）

### 8.1 接口层

- 连接阶段：扫描设备，按序列号绑定目标；记录 `device_type`、手性、固件、硬件版本、IP/端口。
- 状态阶段：同时订阅 `joint_states` 和 `joint_diagnostics`；按 `nid` 转扁平索引；状态帧不足 20 项时进入降级/停机策略。
- 命令阶段：每次发送完整 20 项；位置 rad、速度 rad/s、前馈 effort A；不把 effort 写成未经验证的 N·m。
- 配置阶段：先低 `effort_limit`，再调 `kp/kd`；所有写入拒绝非有限值和负值；记录 SDK 连接日志中的 `kp/kd` 回读。

这些是官方接口事实与直接工程化整理；“降级/停机策略”的具体阈值仍需用户根据真机风险评估确定。来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md)、[关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md)（访问：2026-11-01）。

### 8.2 安全层

- 使能前确认空间无手、人和易碎物；官方示例使用 `kp=3.0`、`kd=0.05`、`effort_limit=1.5 A` 作为保守示例，不应视为所有负载的保证。
- 大幅位置变化必须拆成小步，避免高 `kp` + 大阶跃导致高速运动；接触任务先低电流上限探测。
- 应用层实现命令新鲜度 watchdog：超时停止发布并调用 `emergency_stop()` 或 `disable()`，因为官方 ROS 2 驱动没有 command watchdog；调用后仍需确认关节确实停止。
- 失联按“停机—诊断—清故障/断电—重新使能”处理，不能自动重发上一帧命令。
- 机器人臂侧碰撞检测、速度和工作空间隔离由集成方负责；手侧保护不能替代机械臂安全控制。

前两条直接来自官方控制指南，后三条是根据官方故障/ROS2限制形成的安全设计要求。来源：[控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#parameter-safety-limits)、[维护](https://docs.wuji.tech/docs/en/wuji-hand/latest/maintenance.md#safety-responsibilities-when-running-with-a-robot-arm)、[ROS 2 集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/ros2-integration.md#send-joint-commands)（访问：2026-11-01）。

### 8.3 映射与标定层

- 先区分 Hand 2.1/2.2，再选择对应模型和固件；不能将项目已有 URDF 文件自动视作当前官方版本。
- 对每只真机逐关节做小幅、低电流、有人监护的正负运动，建立 `{finger}_S1..S4` ↔ 模型轴、符号、零点、限位表；官方尚未提供轴序映射，这一步是 REQUIRED 的真机确认。
- 自研屈伸电机手套只应产生抽象的 20 关节目标；手套传感器坐标、手性和力反馈算法由本项目维护，不得借用未公开的官方 SDK 私有协议。
- 姿态重定向可以参考官方 21 keypoints → 20 joint angles 示例，但仍须通过真实手套和 Hand 2 低速验证遮挡、限位、对掌和 DIP/PIP 关系。

## 9. 官方明确的未知项 / 需要用户 SDK 与真机确认

下列内容在公开官方资料中未形成可直接依赖的稳定合同，不能猜测：

1. SDK `S1..S4` 与 URDF/MJCF/USD `J0..J3` 的轴序对应、方向符号、每个序列号的零点偏差。
2. `effort`/`current` 到 N·m、指尖力、接触法向力的转换系数、温度/速度/方向补偿和可信误差范围；官方明确不建议用 2.2 电流判断外力。
3. 2.2 原型长期热行为、满载占空比、降额曲线、可靠性寿命、跌落/振动/冲击承诺。
4. XT30 实际针脚极性图、浪涌/保险/接地/峰值与待机功耗（官方说尚未提供）；不得从外观或第三方资料推断。
5. 当前 2.2 版本的机械安装孔位/法兰尺寸若超出官方 STEP/A3 图纸；应以交付包和真机测量复核。
6. 故障时对离线关节发送 20 项命令的固件行为、是否保持上一目标、各 fault code 的实机触发条件和清除时间。
7. SDK/固件在不同版本组合下的 UDP 重传、时钟同步、最大有效吞吐和命令延迟分布；文档只给接口频率/连接参数，不给实时性 SLA。
8. 官方手套重定向示例的内部滤波、延迟和标定质量是否适用于自研手套；只能使用公开输入输出接口，不复制未公开实现假设。
9. 触觉能力是否存在于实际交付手：2.2 文档主体为“非触觉版原型”，GitHub 发布记录另有“指尖触觉模块”版本说明；必须根据实机硬件批次、固件和 `fingertip_online()`/设备信息确认，不能仅凭型号名称判断。
10. “力反馈”是否包含任何面向操作者的电机反驱力矩安全保证；公开资料没有给出此保证，需向官方索取交付规格或自行增加独立力传感器。

## 10. 建议的首次真机验证顺序

> 本节是基于官方文档的工程流程建议，不是官方验收规范。

1. **识别**：记录序列号（判断 2.1/2.2）、固件、硬件版本、手性、IP 和端口；保留 SDK/设备日志。
2. **无负载检查**：确认 11–13 V 供电、线缆无热插拔、网络静态地址可达、状态灯由蓝色呼吸变白色呼吸；确认 20 关节在线、无 active error code。
3. **低风险轴序实验**：固定 `effort_limit` 低值，逐关节小步位置命令，读取 `joint_states` 的 `nid/position`，建立 SDK ↔ 模型映射，不做大阶跃。
4. **控制参数记录**：读取并保存默认 `kp/kd/effort_limit`；单独改变一个参数，确认 SDK 日志和状态响应；断电重连后确认参数恢复行为。
5. **接管闭环**：先离线/仿真验证 20 项命令，再用自研手套的慢速小动作驱动；监控状态流新鲜度、故障码、温度、电流、通信丢包和在线关节数。
6. **停机演练**：测试正常 `disable()`、软件 `emergency_stop()`、拔网线后的停止与恢复流程；验证应用不会在恢复网络后自动继续发旧命令。
7. **力反馈边界**：只有在增加并标定独立力传感器后才闭环使用接触力；Hand 2 `effort/current` 仅作为设备电流/前馈字段和诊断信号。

## 11. 官方来源索引

- [WUJI TECH Wuji Hand 2 官网产品页](https://www.wuji.tech/en/hand2)
- [Wuji Hand 2 产品介绍（2.2 Beta）](https://docs.wuji.tech/docs/en/wuji-hand/latest/overview.md)
- [控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md)
- [关节状态与控制](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/joint-control.md)
- [力矩与 MIT 参数](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/effort-and-mit.md)
- [数据流总览](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference.md)
- [设备参数与操作](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/device-operations.md)
- [诊断与日志](https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/diagnostics.md)
- [电气与网络集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/electrical-network-integration.md)
- [维护](https://docs.wuji.tech/docs/en/wuji-hand/latest/maintenance.md)
- [用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md)
- [快速开始](https://docs.wuji.tech/docs/en/wuji-hand/latest/quick-start.md)
- [版本识别与兼容性](https://docs.wuji.tech/docs/en/wuji-hand/latest/version-compatibility.md)
- [模型资产与加载](https://docs.wuji.tech/docs/en/wuji-hand/latest/model-integration.md)
- [ROS 2 集成](https://docs.wuji.tech/docs/en/wuji-hand/latest/ros2-integration.md)
- [Wuji SDK 官方仓库](https://github.com/wuji-technology/wuji-sdk)
- [Wuji Description 官方模型仓库](https://github.com/wuji-technology/wuji-description)
- [Wuji Hand ROS2 官方仓库（已归档）](https://github.com/wuji-technology/wujihandros2)
- [Wuji SDK 官方发布记录](https://github.com/wuji-technology/wuji-sdk/releases)

> 以上来源均为官方域名 `wuji.tech` / `docs.wuji.tech` 或官方 GitHub 组织 `wuji-technology`；所有链接访问日期均为 2026-11-01。

## 12. ENCOS 手套官方工作上限核查（2026-11-03）

> **适用边界**：本节区分 Wuji Hand 2、ENCOS 官方灵巧手及自研 vE3.3 外骨骼手套。下列官方产品参数不能仅因共用 ENCOS 协议就套给自研手套。本次只访问静态资料与官网，未连接、扫描或使能硬件。

### 12.1 当前能否直接设置“官方连续力反馈上限”

**不能给当前手套填写一个有官方依据的连续 N·m 上限。唯一首要缺失前提是实际安装电机的确切型号/铭牌（或能绑定该批硬件的厂商交付规格）。** 本地原始 SDK、部署 SDK、左右手 CAD/CSV/URDF 与 Studio 导入包的型号核查未建立这项对应关系；安装件名、URDF `effort` 和 GUI 控件范围均不是电机铭牌或额定规格。

取得型号后，才可把该型号的额定连续力矩、适用温度/散热条件及输出轴侧单位，与实际控制模式对应；若厂商仅给额定电流，还需同型号、同输出侧的有效 `Kt` 才能换算 N·m。核查时的软件试验限额为 `0.02 N·m`，后按用户明确选择改为每raw轴 `5 N·m`；两者都**不是已认证安全值，也不是本次查得的官方额定值**。持续抓握/虚拟弹簧阻尼反馈应依据连续额定能力及其工作条件，不能直接使用短时峰值；若峰值允许时长、冷却时间和占空比没有规格，不应自行补造。

### 12.2 第一方文件确认：MIT 的 ±10 N·m 是编码量程

- 官网[下载中心](https://www.encos.cn/xzzx)直接提供[《ENCOS 灵巧手调试说明书 V2.1》](https://www.encos.cn/filedownload/3057611)。该文件正文第 10 页 §6.1 明确：前馈扭矩字段为 `uint12`，整数 **0–4095** 对应 **−10～+10 N·m**；这说明命令字段如何编码，**不是“电机可连续输出 10 N·m”或额定/峰值承诺**。同页位置和速度字段也有各自编码范围，不能据此推出机械限位或硬件能力。
- 同手册正文第 6 页 §3.1 和第 10 页 §6.1 给出 `motor_current = (Kp·位置误差 + Kd·速度误差 + Torque) / Kt`，并明确不同型号的电流/扭矩范围不同。因此 `Torque` 是前馈项，而非存在非零 `Kp/Kd` 时的总输出力矩上限；只钳制前馈项不能声称钳住所有控制项。来源：[官方灵巧手手册](https://www.encos.cn/filedownload/3057611)。
- [《ENCOS 调试技术手册 V1.20EAP》](https://www.encos.cn/filedownload/3052924)正文第 25–26 页 §9、§9.1.1 进一步区分协议数值与系统限幅，并说明不同型号的协议解析范围有差异；其 EC-A4310-P2-36 示例的前馈编码范围是 **−30～+30 N·m**，不能拿来替换上述灵巧手协议，也不能由此推导当前手套额定值。同文件 §3.4 区分供电母线电流与实际控制的相电流，§6.2.3 将 `Kt` 解释为扭矩与相电流之比。

### 12.3 可核实的官方额定/最大参数：仅作不同硬件对照

| 官方资料中的对象 | 可核实数值 | 对本手套的限制 |
|---|---|---|
| ENCOS 官方灵巧手的“指尖参数” | **额定电流 0.16 A；最大电流 0.7 A**。同手册整手为 **20 自由度/20 关节、EtherCAT**。来源：[V2.1 手册正文第 4 页 §2.1.2](https://www.encos.cn/filedownload/3057611)。 | “最大”不等于连续额定；该表未给可直接换算的额定/峰值 N·m、`Kt` 或最大电流允许持续时间，也未在该表明确电流测量口径。该官方整手与自研外骨骼手套不是同一已确认型号，**不能将 0.16 A 或 0.7 A 写入当前手套配置**。 |
| ENCOS EC-A4315-P2-36 关节模组 | 官网列 **额定扭矩 20 N·m、峰值扭矩 75 N·m**。来源：[对应型号官方产品页](https://www.encos.cn/productinfo/55373.html?templateId=6217)。 | 仅表明厂商按具体型号区分额定与峰值；没有证据表明用户手套安装该模组，**两值均不适用于当前手套**。不能把峰值当持续力反馈上限。 |

官方灵巧手手册正文第 4–5 页 §2.3 还明确：接近峰值、堵转或突然重载时发热很快，温度检测存在滞后，温度保护可能跟不上。因此不能用“有过温保护”替代连续额定能力或穿戴安全论证。来源：[官方灵巧手手册](https://www.encos.cn/filedownload/3057611)。

### 12.4 Wuji 的 2 A 上限属于另一设备、另一单位

- [Wuji 官方控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md#mit-control-mode)明确 `effort` 单位为 **A，不是 N·m**，并明确 `effort_limit` 的设备参数硬上限为 **2.0 A**；超限写入被拒绝且原值不变。这是 Wuji Hand 2 对参数写入的约束，**不是 ENCOS/自研手套的电流或力矩上限，也不是 Wuji 可持续满载的认证值**。
- [Wuji 官方用户须知](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md#thermal)说明原型长期连续工作温升/降额尚未验证，禁止长时间满载或高载连续测试；[电机负载能力和反驱/力矩透明性说明](https://docs.wuji.tech/docs/en/wuji-hand/latest/user-notice.md#motor-load-capacity)说明额定/峰值负载曲线并非可对外承诺的设计依据，且不能用原型关节电流判断外部接触力/力矩。
- 因此，“跟踪误差/电流估计接触 → 虚拟弹簧阻尼 → 手套反馈”只能作为待验证的工程方案，不能改写为官方已支持的接触力闭环；即便算法建立，仍必须另用实际手套电机的连续额定限额约束输出。此句为工程推断，不是官方能力承诺。

**资料可达性说明**：官网页面依赖 JavaScript，下载入口通过浏览器读取；上述两份官方调试手册已取得正文。官网另列[《ENCOS 第二代关节数据手册 V1.6EAP》](https://www.encos.cn/filedownload/3053932)，本次下载超时/限流，未据其未读取内容作任何参数结论。未采用经销商宣传值作为上限依据。
