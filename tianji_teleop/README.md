# Motorhand 手套 → 天机 Wuji 二代手

天机遥操作（tianji_teleop）里用 Motorhand 外骨骼手套代替 MANUS 控制 Wuji 二代手的那部分代码，原样整理到这里。
**只有手的部分**：手套读数 → 重定向 → 发给 Wuji 手的目标，以及手套电机的回零、跟随、力反馈。天机机械臂、PICO、
DAgger 都不在这里。

Motorhand 自己的源码只改了两处，都是把写死的值做成参数、默认值不变（见下文"对 Motorhand 源码的改动"），
Motorhand 自己的程序行为不受影响。

## 目录

| 路径 | 内容 |
|---|---|
| `native/` | C++ worker `motorhand_glove_native`：读手套 → vE3.3 模型 → WujiRetarget → 每秒 200 个目标；手套出力（`glove_control.*`）；`bench.cpp` 量一帧计算多久 |
| `motorhand_glove/node.py` | ROS2 节点，在天机工作区里跑：起 worker，把目标发到 Wuji 手话题，把手的状态转给 worker |
| `motorhand_glove/worker.py` | Python 版 worker（备用，只读手套不出力），重定向在 Motorhand 自己的进程里算 |
| `motorhand_glove/feedback.py`、`viewer.py` | Python 版力反馈（早期离线版本；现在出力全走 C++，天机里只有测试还用它）；MuJoCo 可视化（按天机的对指参数画手） |
| `config/motorhand_glove.yaml` | 天机用的参数，每项注释写了 Motorhand 原值和改的原因 |
| `config/motorhand_glove.upstream.yaml` | 同样的键，全部填 Motorhand 原值，换这份就按原来的参数跑 |
| `config/motorhand_feedback/` | 天机右手的力反馈限额和接触参数（Motorhand 原来的在仓库根的 `config/feedback/`，没动） |
| `scripts/` | `build_`（编 C++ worker）、`setup_`（装 udev 规则 + 只读检查手套）、`view_motorhand_glove.sh`（标零 + 可视化） |
| `udev/` | 让普通用户能打开 CANable2 的 udev 规则 |
| `package.xml`、`setup.py` 等 | ROS2 包的元文件（天机里这个目录就是 `src/teleop_inputs/motorhand_glove`） |

## 和天机怎么对接

- **话题**（`sensor_msgs/JointState`，best effort，深度 10，关节顺序是天机 Wuji 驱动的 `COMMAND_JOINT_NAMES`）：
  - 发布 `/<side>_hand/joint_commands`：手套的目标，和天机原来 MANUS + retarget 发的完全一样，下游不用改。
  - 订阅 `/<side>_hand/joint_states` 和 `/<side>_hand/joint_commands`（开了出力时）：手的状态和它最终收到的目标。
    最终目标和本手套 0.5 s 内发过的某一帧相差不到 1e-6 rad，就算"手在跟这只手套"。
- **节点依赖天机**：`wuji_hand2.joints`（天机 Wuji 二代手驱动的关节表）、`tianji_runtime.resources`（工作区路径）、
  天机 `config/robot.json`（每只手的序列号，力反馈限额文件要核对）。所以节点只能在天机工作区里跑；这里是同一份代码。
- **worker 跑在本仓库的 Pixi 环境里**：节点把 PATH/CONDA_PREFIX 指到 `<root>/.pixi/envs/default`，
  `root` 是本仓库检出（默认在天机工作区旁边）。worker 和节点之间：stdout 每行一条 JSON
  （`ready` / 目标 `{"ns", "q"}` / `status` / `stats` / `event`），stdin 每行一条手的状态或最终目标。
- **天机里启动**：`bash bash/run_hands.sh teleop glove:=motorhand`，或整机 `bash bash/run_robot.sh ... --glove motorhand`。
  用本仓库编的 worker：节点参数 `native_binary` 填 `<Motorhand>/.cache/tianji-glove-native/motorhand_glove_native`。

## 手套出力：三种模式（C++ worker，`native/glove_control.hpp`）

照 Motorhand session 的 enable / step_inference / step_takeover 搬来，用它原样的 EncosDriver、InverseWorker、
JacobianWorker、ForceController、GloveOutputWorker；区别是手的状态来自天机的话题，不是 Motorhand 自己连的 Wuji 手。

- **回零**：worker 启动先回零（Motorhand 驱动没回零前不出任何力），电机以 5 rpm 把手套带到零位姿势，手放松。
  每个 worker 只回一次：电机故障重连后从当前姿势接着出力（`EncosDriver::mark_homed`）。容差 0.3 rad。
- **force（人接管）**：手在跟这只手套时，Wuji 手受阻（电流、停住、落后目标）处，手套电机往回推（接触阻抗）。
  要等 Wuji 先追上手套的目标（各关节 0.06 rad 内保持 0.2 s）才开始，否则驱动回零或交接期间手离目标远又停着，
  会被当成卡住、启动就推满上限。
- **follow（模型推理）**：手在跟别的（DAgger 模型）时，手套位置跟随 Wuji 手（逆解），从当前位置 0.5 s 过渡过去。
  每个电机的推力不超过它的力反馈上限：目标限制在实测位置 ± 上限/刚度以内。
- **idle**：没有新鲜的手状态，或手 0.1 s 内没有任何新目标（没人在控制手：开始前、停止后）——不出力，
  命令过期后电机约 0.1 s 内释放。

## 降延迟：做了什么、为什么

| 问题 | 做法 | 效果 |
|---|---|---|
| Encos CAN 驱动每个适配器每 5 ms 只问一个电机，4 个电机一路，每个电机 20 ms 才读一次 | 时隙做成参数（`EncosDriver::set_slot`），天机用 1 ms | 发出时读数从约 25 ms 旧降到约 4.7 ms 旧 |
| Python 版：重定向（每帧 11~26 次 SLSQP 求值）和 CAN 轮询线程在同一个进程里抢 GIL，回包少一半，最大间隔 66~91 ms，撞上 100 ms 过期 | 重定向放进 Motorhand 自己的独立进程（`worker.py`） | 1 ms 时隙也干净：每个电机 189 Hz，p99 11 ms，最大 22 ms |
| Python 一帧 6~8 ms | C++ worker，直接编译本仓库 `native_control` 的驱动、模型、重定向 | 一帧约 0.45 ms（bench：模型 0.32 ms + 重定向 0.13 ms）；真机右手每秒 200 个目标、读数约 4.7 ms 旧、计算 0.7 ms |

下游天机这边量过不慢：HandGate 即时转发，桥接 5 ms，Wuji 驱动 200 Hz。

## 对 Motorhand 源码的改动（默认值都不变）

- `native_control/encos_driver.{hpp,cpp}`
  - `set_slot()` / `slot()`：CAN 时隙，默认 5 ms。
  - `set_position_kp()` / `position_kp()`：`write_position` 的 MIT 刚度，默认 45 Nm/rad。
  - `mark_homed()`：不动电机、把手套当作已回零（和 `home()` 一样的检查，不占输出）。
  - 电机失败时说明原因：错误码、从没回过、多久没回、读数非有限值。原来只说 "stale, missing, invalid or faulted"。
- `native_control/wuji_retarget.cpp`：对指权重上限从写死的 0.7 改成读重定向 yaml 的 `retarget.pinch_weight_max`
  （默认 0.7；`config/retargeting/official/*.yaml` 里显式写上了 0.7）。vendor 里的 Python 版仍固定 0.7，
  `worker.py` / `viewer.py` 运行时改它。

天机 worker 的 `--pinch-*` 不改 Motorhand 的文件：worker 启动时以官方 yaml 为底、写入这几个值，生成一份
`$TMPDIR/motorhand_glove_retarget_<side>_<uid>.yaml`（`urdf_path` 改成绝对路径），本进程里所有 WujiRetarget
（正向、力反馈的雅可比、跟随的逆解）都读这一份。

## 参数对照：Motorhand 原值 → 天机值

`config/motorhand_glove.yaml`（节点参数 / worker 命令行）：

| 参数 | 原值 | 天机 | 原因 |
|---|---|---|---|
| `slot_ms` CAN 时隙 | 5 ms | 1 ms | 降延迟，见上 |
| `rate_hz` 输出频率 | 50 Hz（Motorhand 的 teleop） | 200 Hz | 天机 Wuji 驱动每秒发 200 次，C++ 一帧 0.45 ms 跟得上 |
| 回零次数 | 每次连上都回 | 每个 worker 只回一次 | 电机故障重连时不把正在用的手拽回零位 |
| 回零容差 | 0.15 rad | 0.3 rad | 戴着时拇指外展停在离零位约 9°，0.15 永远到不了，电机顶着手直到报故障 |
| `follow_kp` 跟随刚度 | 45 Nm/rad | 15 Nm/rad | 一次 DAgger 里 2 号电机（拇指外展）报错误码 1（过热）11 次：被拇指顶开 9°，45 一直出约 7 Nm，15 约 2.4 Nm |
| 跟随推力上限 | 不限 | 力反馈上限（无限额文件时 `--follow-max-nm`，2 Nm） | 跟随时推得不比力反馈狠，防过热 |
| `pinch_d1_cm` / `pinch_d2_cm` | 2 / 4 cm | 3 / 6 cm | 面向任务的接管，更容易捏上 |
| `pinch_weight` | 0.7 | 1.0 | 用户定：d1 以内指尖只对齐拇指尖、不再照手形 |

`config/motorhand_feedback/max-torque-linear-right.json`（原文件 `config/feedback/max-torque-linear-right.json`）：

| 键 | 原值 | 天机 | 原因 |
|---|---|---|---|
| `glove_max_output_torque_nm` | 每个电机 10 Nm | 4 Nm，拇指外展（2 号）2 Nm | 接触感清楚、限制推人的力；2 号同上面的过热 |
| `max_torque_slew_nm_s` | 6 Nm/s | 60 Nm/s | 接触时力来得快、干脆 |
| `wuji_max_current_a` | 2.0 A | 1.5 A | 天机 Wuji 驱动的电流上限是 1.5 A，估力用的电流要和它一致 |
| `jacobian_refresh_interval_s` | 0.02 s | 0.05 s | 当时没有单独记录原因 |
| `wuji_serial_number` | WH2KA01260814006 | WH2KA01260814017 | 限额绑定一只手，worker 会核对；这是天机右手 |

`config/motorhand_feedback/contact-impedance-right.json`（原文件 `config/feedback/contact-impedance-right.json`）：

| 键 | 原值 | 天机 | 原因 |
|---|---|---|---|
| `reflection_gain` | 10000 | 1500 | 接触感更硬更快 |
| `output_filter_hz` | 8 Hz | 25 Hz | 同上 |
| `opening_speed_deadband_rad_s` | 0.02 | 0.8 | 人只是稍微松手（张开速度 0.8 rad/s 以内）时不撤力 |
| `release_dwell_s` | 0.06 s | 0.15 s | 电流短暂掉下去时接触不断 |
| `stalled_speed_rad_s`、`dwell_s`、`current_filter_hz`、`load_weight_scale_a`、`relative_velocity_limit_rad_s` | 0.2、0.04 s、5 Hz、0.5、2.0 | 0.3、0.03 s、10 Hz、0.3、3.0 | 当时只记了整体目标"接触感更硬更快"，没有逐项记录 |

## 用法（在本仓库检出里）

```bash
pixi install --frozen                                         # 一次
bash tianji_teleop/scripts/build_motorhand_glove.sh           # 编 C++ worker 到 .cache/tianji-glove-native/
bash tianji_teleop/scripts/setup_motorhand_glove.sh           # 装 udev 规则（sudo），再只读读几秒手套
bash tianji_teleop/scripts/setup_motorhand_glove.sh --check   # 只检查：多少电机在答、读数多旧、各手指动了多少
bash tianji_teleop/scripts/view_motorhand_glove.sh --zero     # 手套保持零位姿势标零，再开可视化（要显示器）
.cache/tianji-glove-native/motorhand_glove_bench . 600        # 一帧计算多久
```

手套的适配器一次只能一个程序用：跑这些之前先停天机的 hands 窗口。

## 注意

- **天机仓库现在的编译方式和本分支不兼容**：天机的 `bash/build_motorhand_glove.sh` 在 Motorhand 源码副本上打字符串
  补丁（时隙、回零、故障信息、刚度、对指），本分支把这些改成了正式参数，原来的那几行已经不存在。机器上的
  Motorhand 检出切到本分支后，天机那边的编译会在配置阶段报错（增量编译也会重新配置）。切换时用本仓库的
  `scripts/build_motorhand_glove.sh` 编，并把天机节点参数 `native_binary` 指到
  `<Motorhand>/.cache/tianji-glove-native/motorhand_glove_native`。
- 对指参数生成的临时 yaml 每个 worker 一份（`$TMPDIR/motorhand_glove_retarget_<side>_XXXXXX`），worker 正常退出时删掉。
- 标零：Encos 上电后有的轴整体跳约 ±120°。手套断过电就重新标零（`view_motorhand_glove.sh --zero`，会先备份
  标定到 `.calibration_backup/`）。
- Python 版 worker 只读手套，不出力；出力（跟随、力反馈）只有 C++ worker。
- 只配了右手的力反馈限额和接触参数；左手要用，先按左手的序列号做一份。
