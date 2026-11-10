# Motorhand 手套力反馈独立仿真

本目录是从 `hbw1010/Motorhand_Teleop` 抽取的手部链路独立工程，**不包含 Wuji 灵巧手、ROS2、PICO、DAgger**。链路为：

```
真实手套编码器/状态 → 12 电机角 → CAD 闭环运动学 → MuJoCo 手套本体模型
        → 与虚拟物体接触/形变/受力 → 接触力投影到 12 个电机轴
        → 力矩上限/斜率/滤波/爬升 → 真机 MIT 力矩前馈（kp=kd=0）
```

## 1. 目录

| 路径 | 内容 |
|---|---|
| `src/glove_sim/` | 仿真、运动学、物体、SDK、遥测、UI |
| `src/glove_sim/sdk/real.py` | Encos 12 轴真实手套封装，只读检查/使能/力矩前馈 |
| `src/glove_sim/sdk/zero.py` | 软件零位写入（保留相对限位） |
| `config/default.yaml` | 仿真、物体、反馈、遥测配置 |
| `../assets/hand_motor_ve33/hand_motor_ve33_R/` | 手套本体 MJCF、STL、闭环求解脚本 |
| `../hardware/encos_dexhand/` | Encos SDK 与 `glove_mapping.json` / `joint_calibration.json` |
| `../src/motorhand_teleop/encos_home.py` | 原仓库统一使能回零工具（`home-glove` 调用） |
| `../tianji_teleop/udev/90-motorhand-glove.rules` | CANable2 普通用户访问规则 |
| `../config/feedback/` | 原仓库反馈限幅配置（当前仿真采用 `config/default.yaml` 内参数） |
| `scripts/smoke.py` | 无硬件冒烟测试（23 项） |
| `tests/` | pytest 离线测试 |
| `data/` | 默认记录目录 |

## 2. 环境安装

要求：Linux x86_64、Python 3.12（由 Pixi 管理）、可用的 OpenGL 图形会话（仅 UI 需要）。

```bash
cd /home/fcl/hbw/Motorhand_Teleop/sim
pixi install --frozen        # 使用 pixi.lock，避免重新求解卡住
pixi run smoke               # 23 项离线冒烟
pixi run test                # pytest
```

如果 `pixi install` 因网络/锁文件更新很慢，使用 `--frozen`；若 pypi 包被 Pixi 重装后缺失，可修复：

```bash
uv pip install --python .pixi/envs/default/bin/python \
  -e . -e ../hardware/encos_dexhand
```

没有图形显示时所有仿真都必须加 `--headless`；`pixi run sim` 会检查 `DISPLAY/WAYLAND_DISPLAY` 并给出提示。

## 3. 真机设备检查

1. 接 3 个 CANable2（USB `16d0:117e`，slcan，1 Mbps）；同一时间只允许一个程序占用串口。
2. 安装 udev 规则（只做一次）：

```bash
sudo groupadd -f plugdev
sudo usermod -aG plugdev "$USER"      # 重新登录后生效
sudo cp ../tianji_teleop/udev/90-motorhand-glove.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
lsusb | grep -i 16d0:117e
```

3. 只读检查 12 轴：

```bash
cd /home/fcl/hbw/Motorhand_Teleop/sim
pixi run check-glove
```

`check-glove` 打开串口、扫描、读取角度/电流/温度/报警，**不使能、不发送位置或 MIT**。只有 12 个电机全部在线且无报警才返回 `CHECK_OK`。

## 4. 置零 / 标零 / 回零（危险动作分开）

### 4.1 软件零位（不驱动电机）

把当前手套摆到希望的零位，然后：

```bash
# 全手一次性读取当前 12 轴作为零位（自动备份 mapping）
pixi run zero-glove --all

# 或逐指交互：摆好当前手指，按 Enter 确认，输入 q 取消
pixi run zero-glove --finger thumb
pixi run zero-glove --finger index
pixi run zero-glove --finger middle
pixi run zero-glove --finger ring
pixi run zero-glove --finger pinky
```

写入 `../hardware/encos_dexhand/configs/glove_mapping.json` 的 `zero_deg`，同步相对限位并更新 `joint_calibration.json`；每次写入前备份到 `configs/zero_backups/`。**不固化电机硬件零位，不发送位置/MIT**。

### 4.2 统一使能回零（会让电机运动）

此命令调用原仓库 `encos_home.py`，按 `zero_deg` 以 5 rpm 回零，J1 1.5 A、其余 1 A：

```bash
pixi run home-glove --tolerance-rad 0.15 --speed-rpm 5.0
```

> 警告：会立即使能 12 个电机并运动到已标定的零位。手要离开机构活动范围，先做 `check-glove`，确认 `zero_deg` 是本次想要的零位。超时/报警会先失能。

## 5. 仿真与真机遥操作

### 5.1 离线仿真（假手套）

```bash
pixi run sim
# 或自动演示 + 记录，不开窗口
pixi run sim --headless --feedback --duration 6
```

默认假手套会周期开合；`--feedback` 打开的是**仿真反馈**（无真机输出）。`--real` 可让 `sim` 使用真手套，但通常应使用 `teleop`。

### 5.2 真机遥操作

```bash
pixi run teleop
# 无显示器的只读/调试：
pixi run teleop --headless --duration 5
# 确认真机安全后才允许力矩输出：
pixi run teleop --headless --feedback --duration 10
```

UI 启动时反馈默认关闭。先看 `check-glove`、确认手在安全位置、再做 `home-glove`；在 UI 勾选“启用力反馈”后才发送 MIT 力矩前馈。UI 的“急停 / 清零”会立即关闭反馈、把所有电机力矩命令置零并失能（真机路径）；真机读取失败或出现电机报警时运行器也会自动急停。

### 5.3 UI 操作

- 左键拖动物体：物体被 PD 伺服拉向鼠标，**会与手套和其他物体真实碰撞**；拖动速度限幅 15 mm/控制帧，避免快速甩动穿模。选中物体显示为亮黄色。拖动释放后保持当前位置。左键在空白处拖动不改变视角。
- 右键拖动：旋转相机（右键不会选中或移动物体）；滚轮：缩放；F11 或“全屏”按钮：切换全屏，窗口可缩放，面板自适应。
- `Shift + 左键拖动物体`：绕世界 Z 和相机右轴旋转。
- “暂停假手演示”可冻结当前手型，便于摆放物体；“聚焦”把相机对准选中物体；“放到指尖”把选中物体放到指尖可抓取位置；“重置物体”恢复初始布局；“来源与限制”折叠面板列出数据来源。
- “锁定物体”默认勾选：物体保持不动，用于稳定测量。此时可见手型会在接触处按手指**限位**，不会继续穿进锁定物体；反馈力矩仍按未限位的目标手型计算，因此触感保留。取消勾选后物体会被运动中的手套推动/带动（无重力，静止时不会自行漂移）。
- 右侧选择物体，调“大小/硬度”；改变限频 10 Hz 重建，反馈和拖动仍可用。反馈增益滑条只缩放名义力矩，硬上限和急停仍然生效。软物体为青蓝色，硬物体为琥珀色。
- 底部页签：每指仿真接触力、每指最大名义反馈力矩、12 仿真关节位置、12 关节速度、电流（真机实测/假手套代理）、力—接触穿透/柔体形变、传感器时延。
- “开始记录/停止记录”手动控制；急停按钮始终可用。

## 6. 记录、回放与数据含义

```bash
pixi run record --record data
pixi run record --headless --feedback --duration 30 --record data
pixi run replay data/glove_session-YYYYMMDD-HHMMSS.jsonl
```

每次记录生成同名 `.csv` 与 `.jsonl`。CSV 头部注释、JSONL 每行都带来源标记。关键字段：

| 字段 | 含义 / 来源 | 单位 |
|---|---|---|
| `timestamp_mono_ns` | 主机单调时钟时间戳 | ns |
| `timestamp_iso` | 记录写入时的 UTC 墙钟时间 | ISO 8601 |
| `source` | `fake` / `real` / `replay` | — |
| `data_validity` | `simulated` / `mixed`（真机为实测+仿真） | — |
| `sensor_timestamp_ns`, `sensor_age_ms` | 真机为最旧一帧 CAN 反馈的 `received_ns` 与本行时刻之差；假手套为本地生成 | ns / ms |
| `contact_force_<finger>_n`, `contact_force_palm_n` | **MuJoCo 仿真接触力**（不是传感器实测） | N |
| `penetration_<finger>_m` | 接触穿透深度，**不是物体应变**；锁定模式的数值来自未限位目标（用于反馈），可见手型另有防穿模限位 | m |
| `max_object_disp_m` | 柔性晶格顶点的体局部最大位移；仅 `ball_soft` 非零 | m |
| `real_joint_i_rad` | 编码器经 `glove_mapping.json` 零位/方向/比例换算后的模型电机角（真机）/假数据 | rad |
| `sim_joint_i_rad`, `sim_vel_i_rad_s` | 仿真手套对应电机轴位置/速度 | rad, rad/s |
| `torque_raw_i_nm` | 接触力雅可比转置得到的**未限幅名义力矩** | Nm |
| `torque_cmd_i_nm` | 经上限、斜率、滤波、爬升、死区后的**名义力矩指令** | Nm |
| `measured_current_i_a` | SDK 回读电流（真机）；假手套为 `torque/0.035` 的演示代理 | A |
| `estimated_force_i_n` | 无真实力传感器、无力标定，**留空并标记 uncalibrated** | — |
| `grasp_hold` | 任一手指接触力 > 0.2 N 的派生保持指示 | bool |
| `slip_validity`, `slip_m` | 物体默认位姿锁定，**滑移不可用**；留空 | — |

**严格区分**：

- 仿真接触力 ≠ 实测力；CSV 不把它写成 measured force。
- `torque_cmd` 是名义 Nm，不是实测力矩；`torque_sign` 默认全 +1，**方向未在真机标定**。
- `measured_current` 是实测电流，但没有 Kt/电流—力矩标定，不能换算成“真实接触力”。
- `estimated_force` 未标定，不提供数值。
- 物体默认锁定，所以不提供物体滑移测量；`grasp_hold` 只是接触力派生指示。

## 7. 实现要点与限制

- 手套本体模型为 `hand_motor_ve33_R` 的 68 关节闭环结构；直接对 12 电机加闭环扭矩在 MuJoCo 中数值不稳定，因此采用 CAD 闭环解算器做**运动学**驱动，接触仍由 MuJoCo 求解。电机力矩反馈通过闭环约束的隐函数导数（`tree_derivative`）投影，列顺序已固定为 J1..J12。
- 6 个物体：球/方块/圆柱各软硬一对。`ball_soft` 是四面体 flex 晶格并可真实形变（表面细分 1 级，避免低模球体穿模观感）；`box_soft`、`cylinder_soft` 是**软接触刚体**，不产生网格形变。物体默认位姿锁定；鼠标拖动使用碰撞感知的 PD 伺服，物体之间以及物体与手套之间开启碰撞。
- 手套本体有 17 个碰撞代理（球/掌心盒），覆盖掌、MCP/PIP/DIP 和拇指；物体与物体、物体与手套均碰撞。
- **防穿模限位**：手套仍是运动学驱动，但每个控制周期会在私有 `MjData` 中评估“未限位目标手型”的接触；若目标会穿入锁定物体，则按手指冻结可见手型的运动，姿态保留在接触前。锁定模式下 `glove.clamp_soft_allow_m=0`，软/硬物体都要求**可见穿透为 0**；软球的可见网格形变因此只在取消物体锁定后出现。反馈力矩仍从未限位的幽灵接触计算，避免限位把触感抹平。相关配置：`glove.collision_clamp`、`glove.clamp_soft_allow_m`。取消锁定后不做该限位，手套可推动并挤压物体。
- 软体采用 MuJoCo flex 边约束；`ball_soft` 的形变量以 `max_object_disp_m` 记录。大幅挤压或超大尺寸（40–70 mm）会产生较大接触力；名义力矩有硬上限。
- 反馈（默认 `glove.feedback_mode: driver_impedance`，代码见 `glove_sim/impedance.py`）：真实手指按进物体后，仿真手停在表面（代理），**弹簧和阻尼放在 Encos 驱动器内闭环**——MIT 目标角=表面角、`kp`=由物体硬度得到的刚度（5–60 Nm/rad）、仅在继续往里按时 `kd`=0.004·kp；目标角以“当前原始角+偏移”下发，模型角限幅不会变成假误差。驱动器 PD 无力矩上限且双向，所以目标被钳在 `0.85×torque_limit/kp` 以内（弹簧饱和），手指退出时提前一个下发周期（40 ms）把 kp 清零，避免把手指拉回表面。空中只发摩擦补偿前馈 `Fc·tanh(dq/vs)`（1°/s 速度死区，默认 0.25 Nm，J2 0.1 Nm，UI“摩擦补偿 ×”可调）。旧的外置弹簧（Python 侧算力矩，每电机 ~50 Hz 更新）保留为 `proxy_spring`，UI“反馈方式”可切换对比。
- 外置弹簧/旧模式：`kp=0,kd=0` 的 MIT 力矩前馈，默认上限 `[4,2,4,4,4,4,4,4,4,4,4,4] Nm`，斜率 60 Nm/s，滤波 25 Hz，爬升 0.15 s，死区 0.01 Nm。命令 TTL 0.1 s，过期后 SDK 不再重发旧目标。
- UI 使用 DearPyGui + MuJoCo EGL 渲染子进程（隔离 GL 上下文）；自定义深色控制台主题、字号/卡片/状态色，渲染 1280×720（16:9）。曝光已调低（headlight ambient 0.16、diffuse 0.24，key/fill 环境光同步降低），避免白手套和亮色物体过曝。不再绘制低分辨率阴影（会严重锯齿），地面使用程序生成的参考网格；动态纹理直接传 numpy 缓冲区。状态栏同时显示**反馈接触**（幽灵目标，用于触感）与**可视穿透点/深度**（真实主数据），锁定模式下后者应为 0 点/0.0 mm。物理仍按 1 kHz 步进，UI 每帧最多补 30 步；实测抓握/软球接触约 12–20 Hz，暂停演示且无接触约 27–43 Hz。软球 flex 接触是主要计算开销；无图形环境必须 `--headless`。

## 8. 已验证（本机离线）

- `pixi install --frozen`、`pixi run smoke`（23/23）、`pixi run test`（9/9，含防穿模回归测试）。
- 模型可编译；17 个碰撞代理；初始布局无接触；位姿锁定无漂移；拖动位姿保持。
- 软球实际形变（脚本挤压下约 9.5 mm，初始应变严格为 0），硬球无形变；软/硬接触力有限且不同。
- 物体—物体、物体—手套碰撞开启；拖动伺服在自由空间误差 0，向其他物体/手掌拖动时会被接触挡住，不再直接穿过。
- 反馈力矩上限、斜率、滤波、死区路径；闭环雅可比列顺序测试。
- 遥测列、CSV/JSONL 写入、来源/单位/不可用项标记；回放。
- 图形 UI 数秒级连续运行、拖动、滑条重建、反馈开关、急停、记录按钮（本机 X11 实测）；全屏 F11、自适应布局、选中物体黄色高亮、暂停演示/聚焦/放到指尖/重置物体按钮均实测。

## 9. 仍需真机验证（未宣称已验证）

1. 3 个 CANable2 的 USB 识别、端口顺序和 12 轴在线；udev/plugdev 权限。
2. 现有 `zero_deg` 的方向/限位是否正确；`zero-glove` 写入后读数是否一致。
3. `home-glove` 的 12 轴回零方向、速度、报警与急停行为。
4. ~~每个电机的力矩符号~~ 已验证（2026-11-10，手套未佩戴）：12 轴正 MIT 力矩都使原始角增大（`torque_sign` 全 +1）；12 轴 MIT 位置坐标与上报角一致，MIT 位置跟踪方向正确；电流约 0.2 A/Nm；kp≤60、kd≤0.3 驱动器 PD 无振荡；动摩擦 0.6–1.0 Nm（5–10°/s），静摩擦 ≥0.4–1 Nm，部分轴在单方向 >2.5 Nm（平放时受桌面/限位影响）。仍需**佩戴**确认反馈“顶手”的手感与刚度/摩擦补偿取值。
5. 反馈增益/上限/斜率在真实佩戴下的力度和稳定性；是否需要按手指降低。
6. 编码器/CAN 实际时延、掉线、断链保护；命令 TTL 是否足够。
7. 软球在解锁后挤压时的视觉形变是否自然（锁定模式为保证零穿透不显示可见形变）；软方块/圆柱目前只有接触柔顺，没有网格形变。
8. 无真实力传感器，不能验证“实测接触力”；只能验证仿真力到名义力矩的映射与佩戴感受。

## 10. 故障排查

| 现象 | 处理 |
|---|---|
| `未发现 CANable2` | `lsusb | grep 16d0:117e`；查 udev、plugdev、重新插拔；确认串口没有被其他程序占用 |
| `必须 12 个电机在线` | 检查三个适配器、CAN 线、供电；先 `check-glove` |
| UI 无窗口/`没有可用的图形显示` | 在桌面会话运行；或加 `--headless` |
| `pixi run` 卡住 | 用 `pixi install --frozen`，然后 `pixi run --no-install <task>` |
| 真机反馈方向不对 | 立即急停；逐轴修改 `glove.torque_sign` 后重测，不要继续佩戴大力矩 |
| 软球/大物体穿透大 | 降低滑块尺寸或硬度；这是运动学手套被指令穿过物体时的已知限制 |

安全边界：本工程不替代硬件急停/断链保护；首次真机测试必须空载、低增益、有人看守。
