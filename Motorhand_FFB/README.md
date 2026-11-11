# Motorhand_FFB

独立的 Motorhand 外骨骼手套到 Wuji Hand 2 的手部遥操作项目。范围只覆盖手：

- 12 电机手套读取、vE3.3 闭链和 WujiRetarget；
- Wuji Hand 2 目标发布；
- 手套回零、跟随和接触力反馈；
- C++ 实时 worker、Python 只读备用 worker、ROS2 话题适配器和 MuJoCo 只读可视化。

项目包含运行所需的 Motorhand 核心、WujiRetarget 资产、Encos 驱动和配置，不依赖兄弟目录 `/home/boen/hand/Motorhand_Teleop` 或天机工作区。ROS2 适配器使用标准 `sensor_msgs/JointState`，因此仍可接入天机的 Wuji 手驱动，但不要求安装 `wuji_hand2`、`tianji_runtime` 或读取天机 `config/robot.json`。

## 目录

| 路径 | 用途 |
|---|---|
| `src/motorhand_teleop/` | 手套运动学、重定向、控制、反馈和 native_control 实现 |
| `tianji_teleop/native/` | 手部 C++ worker：手套读取、重定向、跟随和力反馈 |
| `tianji_teleop/motorhand_glove/` | ROS2 节点、Python 备用 worker、反馈和可视化 |
| `config/motorhand_feedback/` | 右手现有参数、左手模板及说明 |
| `hardware/encos_dexhand/configs/` | 手套轴映射、零位、方向和软限位 |
| `config/retargeting/official/` | WujiRetarget 左右手配置 |
| `vendor/wuji-retargeting/` | 固定版本 Wuji Hand 2 重定向与模型资产 |

## 安装与构建

Pixi 环境固定使用 PyPI 的 `mujoco==4.0.0`；不要把 conda 的旧版 MuJoCo 混入该环境。

```bash
cd /home/boen/hand/Motorhand_FFB
pixi install
pixi run validate-assets
bash tianji_teleop/scripts/build_motorhand_glove.sh
```

构建产物：

```text
.cache/tianji-glove-native/motorhand_glove_native
.cache/tianji-glove-native/motorhand_glove_bench
```

只读检查手套连接，不使能电机：

```bash
bash tianji_teleop/scripts/setup_motorhand_glove.sh --check
```

## ROS2 接入

先构建 `tianji_teleop/` 目录中的 ROS2 Python 包，再使用参数文件：

```bash
colcon build --base-paths tianji_teleop --packages-select motorhand_glove
source install/setup.bash
ros2 run motorhand_glove motorhand_glove --ros-args \
  --params-file tianji_teleop/config/motorhand_glove.yaml
```

节点协议：

- 发布 `/<hand>/joint_commands`：20 维 Wuji Hand 2 目标；
- 订阅 `/<side>_hand/joint_states`：实际手状态；
- 开启力反馈或手套跟随时，订阅 `/<side>_hand/joint_commands`：最终目标；
- C++ worker 通过 stdin/stdout JSON 行协议与 ROS2 节点通信。

节点默认使用本项目根目录。也可以给 ROS2 参数 `root`，或设置 `MOTORHAND_FFB_ROOT`。

## 力反馈模式

`config/motorhand_glove.yaml` 默认启用右手的：

- `force_feedback: true`：Wuji 手受阻时，手套提供名义力矩前馈；
- `follow_hand: true`：其他控制器主导 Wuji 姿态时，手套跟随手部姿态；
- `home_glove: true`：worker 首次启动时回零。

右手参数文件绑定 Wuji 序列号，并限制名义输出上限。电流、SDK 名义力矩和软件估计都不是实测接触力；首次上手必须低负载、确认方向和急停后再运行。

左手只提供模板：

```text
config/motorhand_feedback/max-torque-linear-left.template.json
config/motorhand_feedback/contact-impedance-left.template.json
```

使用左手前必须替换 Wuji 序列号，并重新验证电流上限、方向、阻抗和接触阈值。不要直接把模板当作已标定参数。

## 只读可视化与标零

```bash
bash tianji_teleop/scripts/view_motorhand_glove.sh
bash tianji_teleop/scripts/view_motorhand_glove.sh --zero
```

`view_motorhand_glove.sh` 的可视化不连接 Wuji 真机、不发送 Wuji 目标；`--zero` 会修改本项目手套标定文件并在 `.calibration_backup/` 生成备份。

## 离线验证

```bash
pixi run validate-assets
pixi run -e hardware test-real-hand-native
pixi run -e hardware test-control-native
```

这些检查不等于人体穿戴和真实接触力验收。真实设备验证仍需单独确认手套轴映射、Wuji 序列号、限位、急停和左右手参数。

## 手套本体 MuJoCo 交互仿真

`sim/` 直接使用 vE3.3 手套闭链 CAD 模型；它不接入 Wuji 手、Wuji SDK 或 WujiRetarget。真实手套输入经过现有标定与闭链求解后，作为 MuJoCo 虚拟手套的电机位置目标。虚拟手套由有限力矩伺服驱动，物体接触会阻挡其运动；接触力在虚拟姿态下通过接触点雅可比和闭链虚功映射到模型电机的名义力矩。

```bash
cd /home/boen/hand/Motorhand_FFB
pixi install

# 无硬件：周期性脚本动作、MuJoCo 窗口、本机实时曲线和退出后的 PNG
pixi run python -m sim run --scenario press --duration 15 --plot

# 可形变橡胶球：中心与整体姿态受约束，球体仍可变形
pixi run python -m sim run --scenario grasp --object sphere --material rubber --object-size 0.05

# 较大的固定方块或圆柱；每次运行选择一个物体
pixi run python -m sim run --scenario grasp --object box --material wood --object-size 0.06
pixi run python -m sim run --scenario press --object cylinder --material foam

# 无窗口离线运行；显式启用带重力和支撑平面的自由物体
pixi run python -m sim run --headless --duration 10 --scenario grasp --object-motion free --plot

# 真实手套只读：不回零、不使能力控、不发送电机出力
pixi run python -m sim run --input glove --scenario press
```

图形模式启动后终端显示仅绑定 `127.0.0.1` 的实时曲线 URL，在本机浏览器打开；`--no-live-plot` 可关闭服务，`--headless --live-plot` 可在无 MuJoCo 窗口时启用它。曲线显示使用有界采样，完整记录以 JSONL 为准。日志默认写入 `recordings/sim/`，也可用 `--log` 指定新文件；已有文件不会覆盖。日志退出后可单独绘图：

```bash
pixi run python -m sim plot recordings/sim/你的日志.jsonl
```

`--object sphere|box|cylinder` 分别选择球、方块和圆柱；默认物体的直径或边长为 0.05 m（原球直径为 0.03 m）。`--object-size` 以米指定球直径、方块边长，或圆柱的直径及高度。`--object-position X Y Z` 以米指定物体中心的世界坐标；未指定时按初始手套姿态放置。橡胶、泡棉**球**使用三维弹性体；木材、金属球以及所有方块、圆柱使用刚体。默认模式固定可形变球的中心节点，并通过三个物理约束抑制整体转动，表面仍可变形；`--object-motion free` 不施加这些固定约束，物体可整体移动或旋转，并受重力和支撑平面影响。

| `--material` | 外观 | 默认软硬预设 | 摩擦系数 |
|---|---|---|---|
| `foam` | 蓝色 | `soft` | 0.9 |
| `rubber`（默认） | 绿色 | `medium` | 1.1 |
| `wood` | 棕色 | `hard` | 0.65 |
| `metal` | 灰色高光 | `hard` | 0.4 |

柔性球资产位于 `sim/assets/deformable_ball.xml`，可由 MuJoCo 4.0 独立加载；场景启动时根据物体尺寸、材质、软硬和固定方式配置该资产。它是由三维 [flexcomp](https://mujoco.readthedocs.io/en/4.0.0/XMLreference.html#body-flexcomp) 展开的低层 `flex`，以 27 个插值节点驱动 99 个网格顶点；顶点运动由物理接触和[体积弹性](https://mujoco.readthedocs.io/en/4.0.0/XMLreference.html#flex-elasticity)计算，不是渲染缩放。固定模式的三个绳索等式约束抑制球体材料坐标系的整体转动，不锁死局部形变。软、中、硬分别使用 10000、15000、30000 Pa 的未标定杨氏模量，`--stiffness soft|medium|hard` 可覆盖材质预设；这些数值与摩擦系数都不是实测材料属性。当前网格仅用于实时交互演示，不能等同于经过验证的橡胶变形或应力仿真。若四面体相对初始体积降到 2% 或发生翻转，仿真会报错并停止，避免继续输出无效接触力。

输入手套与仿真手套不是同一个状态：碰撞阻挡时两者可能不一致。固定柔性球的中心与整体转动受到约束，但表面和顶点质心仍会随受压移动；数值求解也可能存在微小约束误差。MuJoCo 接触是软约束，仍可能出现有限穿透。涉及手套的碰撞目前仅启用五指零件与物体之间的接触；手套基座的碰撞以及手套零件间自碰撞未启用。

图形窗口可通过三轴尺寸滑条和鼠标编辑当前物体，修改会应用到真实 MuJoCo 碰撞与弹性模型，而非只改变显示。尺寸为物体局部 X/Y/Z 方向的全尺寸；三轴不等时，球变为椭球、圆柱变为椭圆截面圆柱。松开滑条后会重建物体：原有形变和物体速度清零，仿真手套状态和仿真时间保留。默认固定物体在编辑后仍保持固定；`free` 物体仍可受重力运动。无真实力控输出时按 `R` 可重置自由物体。形状和材质仍需通过启动参数选择。

真实力输出开启时，可以左键拖动物体**位置**；按下物体后程序先请求手套退出力矩输出并发送 0 A 状态探测，再移动仿真物体。拖动和等待恢复期间没有接触力反馈，仿真手套进入编辑离合模式：虚拟电机速度目标限于 0.5 rad/s，虚拟驱动力矩限于 0.05 Nm；有物体接触时不累积未跟随的输入目标，避免暂停期间持续挤压柔性球。无接触后仿真姿态会逐步追赶输入。这些参数只约束**仿真电机**，不是手套电机的力矩或人体安全限额，也不改变球的材料参数。

松开鼠标后必须等物体到位、各仿真接触点的法向力不超过 0.01 N 且穿透不超过 1 µm，并且所有虚拟电机与手套输入的最大位置误差不超过 0.05 rad，程序才请求恢复同一会话的力控，不会再次回零。如提示 `OPEN GLOVE`，请松开手指；仍无法对齐时把球移离手套。真实输出时尺寸滑条、姿态旋转和 `R` 重置仍禁用。该暂停是软件请求，并未独立验证电机实际电流或受力为零，不能替代物理急停；网格塌缩保护仍会在检测到异常时停止仿真。

为防止一次鼠标跳动把柔性网格直接推穿手套，编辑位置和姿态按每个仿真周期最多 1 mm 的最远点位移逐步应用，并检查实际碰撞；进一步增加过深穿透的移动或尺寸会被拒绝，窗口显示 `Blocked: move away from glove`。先把物体移离手套，再调整尺寸或姿态。该限制用于仿真数值稳定，不是人体安全保护。

无真实出力及拖动物体暂停输出时使用固定仿真步长，计算跟不上时仿真会变慢，不会用更大的步长追赶；实时曲线页显示近 50 帧仿真倍速，JSONL 分别记录实际 `control_interval_s` 和 `simulation_step_s`。真实力输出活动期间仍使用实际周期并保留 100 ms 超时保护；这些机制不保证硬实时。

操作方式：

- 物体上按住鼠标左键拖动：沿当前画面平面移动；切换视角可调整其他方向。
- 无真实出力时，物体上按住 `Shift` + 左键拖动可旋转姿态；红、绿、蓝箭头分别表示物体局部 X、Y、Z 轴。
- 无真实出力时，右侧 X/Y/Z 滑条分别调整三轴全尺寸，默认范围 0.01–0.20 m；释放鼠标后生效。
- 空白处左键拖动旋转视角，右键拖动平移视角，滚轮缩放。

**真实力反馈默认关闭。碰撞动力学更改后尚未完成真机力反馈验收，暂勿启用 `--execute`。** 首次硬件验收必须有人监护，确认急停、手套佩戴位置、零位、轴方向、限位以及低负载试验条件。开启力控会先执行实际回零，不能在不允许手套运动时运行。只有明确传入 `--input glove --execute --acknowledge-homing --raw-torque-limit-nm <名义限额> --ports <CAN1> <CAN2> <CAN3>` 才会启用；三个适配器必须由操作者确认属于同一只手套。限额必须针对设备确认，程序要求 0.01–0.2 名义 Nm，此范围并非人体安全认证。`--gain` 默认为 0.1，`--slew` 默认为 1 Nm/s；手套源状态的 100 ms 租约、计算超时、设备错误或程序退出均会触发停止/失能流程。程序不是硬实时系统，也不能替代独立急停。

真实左手必须通过 `--side left --mapping <左手映射文件>` 显式选择单独标定的手套映射，文件须包含 `"hand": "left"`；不能沿用默认右手映射。只读模式如不指定 `--ports`，会由现有 SDK 自动发现串口，应先确保不会连接其他设备。

日志区分输入电机/指节位置和受碰撞约束的仿真电机/指节位置、MuJoCo **仿真接触力**、限幅前请求力矩、限幅后名义力矩、实际向 SDK 提交的**名义指令**和设备返回的**原始电流**；缺失值为 `null`。`feedback_pause_reason` 和 `feedback_transition` 记录拖球暂停和恢复请求；暂停期间周期力矩指令为 `null`，恢复力控时底层另有接近零值的初始化帧，不能把该字段当作已测物理力为零。`virtual_servo_mode`、`virtual_servo_limit_nm` 和 `input_alignment_error_rad` 分别记录虚拟伺服模式、仿真电机力矩限制及输入与仿真电机的最大位置误差，不代表真实电机的力矩。

`object_dimensions_m`、`object_position_m`、`object_orientation_wxyz`、`object_edited` 和 `object_edit_blocked` 按帧记录物体尺寸、位姿和编辑结果；柔性物体位置与姿态分别由顶点质心和最佳拟合材料坐标系估计。柔性球额外记录 `object_deformation_m`：顶点相对顶点质心的径向距离变化最大值，刚体该值为 `null`；曲线中的球体形变和接触穿透分别显示，不能混为一谈。反馈力矩按虚拟手套的位置计算；接触阻挡使输入和虚拟姿态不同时，不能把投影结果当成实测力或已校准的人体接触力。实时界面与 `torque_limiting.png` 标注受限轴数。滑移为接触点切向相对速度；多指接触/滑移状态是启发式指标，不等于经过验证的抓取保持。阶段延迟是同一单调时钟下的软件时间差，并非物理电机响应延迟。

当前手套基座固定，未接入腕部六自由度位姿跟踪，不能通过移动真实手腕移动或抬起虚拟手套。模型的碰撞表面是 CAD 网格凸包，并非人手皮肤；接触力不是传感器实测值，电流和名义指令也不是标定力估计。无独立测力传感器和硬件标定时，不能据此声称真实反馈力准确。真实手套联机、人体触感、稳定夹持和滑移判据仍须现场验收。
