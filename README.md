# WujiHand2 Force Glove

自研 vE.3.3 屈伸电机手套驱动 WujiHand2 的独立研究项目。目标是先完成可验证的姿态接管，再在完成单机标定和安全验收后引入有限力反馈。

## 当前方案

手套的12个运动学独立坐标先经过真实闭链模型生成21个 canonical 任务点，再交给官方
`wuji-retargeting` 的 Hand2 优化器输出20个语义关节目标。项目不维护手写12→20关节映射。

真机边界的控制量必须分开：Encos 原始编码器 degree 经 `zero/sign/scale` 变成
12 个模型电机角，再由 CAD 机构 FK 得到 `GlovePose` 的12个真实指节 rad。
反向位置命令先由真实指节做机构 IK，换回原始电机角并执行软限位；发生饱和后
重新 FK。配置中的 `joint` 是电机轴绑定标签，不表示电机角等于同名指节角。

四指 MCP 侧摆在手套侧没有传感器。当前第一版把四个 Hand2 MCP 侧摆关节通过
`retarget.fixed_joints` 硬约束为0，正向和逆向求解都不参与侧摆；恢复侧摆前需要可观测输入和独立标定。

官方资料显示 WujiHand2 的 `joint_states.effort` 和 `joint_diagnostics.current` 单位是 A，官方没有为2.2原型承诺可直接换算外部力矩或指尖接触力。项目默认拒绝未标定的触觉输出。

## 控制链路

```text
手套12维独立角
  -> vE3.3闭链求解
  -> 21个任务点
  -> 官方 WujiRetarget
  -> Hand2 URDF语义20维目标
  -> WujiDriver

Wuji effort/current
  -> 单机电流-力矩标定
  -> 完整重定向雅可比 J_retarget.T
  -> 手套12维力矩
  -> vE3.3传动雅可比
  -> 手套电机电流
```

正常位置接管只运行一次重定向；`J_retarget` 为低频中心差分估计，不能把直接关节映射当作其替代品。

## 双向任务空间与无扰切换

`bidirectional.py` 提供共享 canonical 21点任务空间：

```text
GloveTaskModel  -> HandTaskState -> WujiHand2Retargeter
WujiHand2TaskModel -> HandTaskState -> GloveInverseSolver
```

两条方向可以计算候选目标，但 `ControlMode` 只允许一个位置主导方发送命令。
`HandoverController` 以切换瞬间的当前命令为锚点，用平滑过渡避免更换求解方向时位置跳变。
`BidirectionalStateBus` 只保存两侧最新状态和任务状态，不直接发送硬件命令。

## 真机 WujiHand2 接入

两侧真机实现集中在 `src/wujihand2_force_glove/real_hand/`：

- `wuji_hand2.py`：本地 WujiHand2 20关节位置驱动。
- `motor_hand.py`：Encos 原始电机与真实指节的机构转换、回零和 MIT 输出。
- `control.py`：外部推理 q20 同时驱动两侧，以及手套接管的控制周期。
- `runtime.py`：真实反馈双向跟随、单主导切换和命令入口。
- `native/`：从 DexGlove 移入的 C ABI 安全核心与官方 C SDK 运输层。

不再导入 DexGlove 的 `real_hand`、`HandPose` 或骨架生命周期。Python SDK 固定
`2026.8.31`；C SDK 同版下载有 SHA256 校验，原生库及 SDK 都在本项目 `.cache/`。
首次构建需要 CMake 和 C++17 编译器：

```bash
pixi install -e hardware
pixi run -e hardware build-real-hand
pixi run -e hardware wuji-discover
pixi run -e hardware real-hand inspect \
  --wuji-serial WH2KA01260814006 --hand right --glove-hand right
```

以上均不使能、不回零。Encos 读取反馈的协议轮询包含 0A 伺服探针，
并非 CAN 总线完全不发报文；不要与 GUI／另一控制进程同时占用同一手套。
示例序列号及固件 `2.2.3` 来自现场只读识别，更换设备后必须重新检查并批准。

**固定实控配置：手套 → Wuji2**。右手，设备 `WH2KA01260814006`，批准固件 `2.2.3`，
目标频率50Hz，Wuji MIT `kp=5.0/kd=0.1`。只向 Wuji 输出位置，手套只读，不使能或回零。

```bash
pixi run -e hardware wuji-teleop
```

该任务会实际使能并控制 Wuji，不是仿真。启动过渡保持0.5秒，运行至 `Ctrl+C` 请求停止；
零位、方向和软限位继续读取已有标定文件，不冻结标定数值，也不修改手套增益。
通用 `real-hand follow` 的正向目标频率默认50Hz；反向仍默认20Hz，不随本次固化改为50Hz。

**Wuji 实测姿态带动手套**：Wuji 只读，不向它发 hold，也不代替外部推理。
首次取得手套输出权先安全回零，正常输出为 `kp=45/kd=0.5` 的 MIT 位置命令。

```bash
pixi run -e hardware real-hand follow \
  --mode hand2-to-glove --wuji-serial WH2KA01260814006 \
  --hand right --glove-hand right --approved-firmware 2.2.3 --rate 20 --execute
```

去掉 `--execute` 并加 `--duration 3`，可用两侧真实反馈做只读计算试跑。
CLI 在取得输出前预热求解器，冷启动帧只作数值种子；每个正式周期重新采样、
重新求解，保留真实接收时刻和原始100ms期限，超期拒发，不续期旧输入。
执行模式保持20Hz或更高；频率过低会让原始命令在下一周期前过期，
SDK转为0A探针而非永久保持旧MIT目标，不能靠延长输入期限掩盖。
退出只释放本会话输出，不把只读输入端失能；不额外触发退出回零。

`PairRuntime.request_mode(FollowMode...)` 提供切换 API，不实现按键界面。
新跟随端从实测姿态五次平滑接入；同会话后续切换不重复回零。
正向求解四指侧摆仍固定0，但实测侧摆不伪装成0，切换时平滑归零。
两侧手型分别用 `--hand` 与 `--glove-hand` 指定，不用左 CAD 冒充右手套。

Wuji2 控制默认采用 MIT `kp=5.0/kd=0.1`，在显式使能时写入；
只读连接／检查不写参数。设备重启会恢复固件默认，后续用本项目启动控制会重新应用这组值。
手套的位置跟随仍为 `45.0/0.5`，不混用两侧增益；Wuji 的电流限值本轮不修改。

外部推理同时驱动两侧仍使用 `RealWujiControlLoop.step(inference_positions_rad=q20)`；
接管使用 `request_mode(ControlMode.GLOVE_TAKEOVER)`。Hand2 从已发命令锚点平滑，
手套从自己的实测模型电机 m12 锚点五次平滑接入：最终 Hand2 目标经同一机构
逆解得到完整 CAD 树，复用树中的电机目标插值，再由 motor FK 生成实际 q12。
这不是 Hand2 关节到电机的线性映射；外部 `GlovePose` 和 `tick.shadow` 仍是真指节 q12。
不再对每帧过渡后的 q12 做一次重复机构逆解，也不拿预热帧或上帧树冒充本次目标。
仅在可行域内保证过渡连续；实测角度若已越软件界，发出的目标仍必须限幅，限位优先。
两个 SDK 不提供跨设备原子事务，发送途中硬件故障不能伪称两侧可原子回滚。

共享 CAD 的 FK、闭链点残差和解析雅可比由 C++17 几何核计算，
`pixi install` 会在安装本项目时编译；C++编译器由 Pixi 管理。
它只替换相同几何的 Python 热点，正逆解、目标函数、约束和参数不变，
不构建第二套映射算法；未编译时需重新安装，不静默回退慢路径。

当前 Wuji C ABI 6 从同一 `joint_states` 包按NID重排位置、速度(rad/s)和实测
effort电流(A)，保留真实收包与设备时戳。可选量缺失或非有限时返回 `None`，不填假零。
逐帧命令仍仅发位置；非零速度／effort命令明确拒绝，不因新增反馈而静默支持力矩输出。

### 标定后的力反馈接管

现有 `wuji-teleop` 不变：手套只读，Wuji位置跟随，不等于力反馈接管。
显式提供 `--feedback-calibration` 后，手套成为唯一位置主源，但自身也使能输出反作用力矩：

- Wuji位置控制保持 `kp=5/kd=0.1`；推理主导时手套位置跟随保持 `45/0.5`。
- 人手接管的手套使用独立MIT纯前馈 `kp=0/kd=0`，不追踪Wuji逆解位置或保持旧手位。
- Wuji电流按Wuji自己的20轴带符号关节侧等效Nm/A（含内部减速）和偏置换算执行器力矩；
  手套另用与其SDK原始角坐标对偶的12轴Kt、电流上限和斜率。两侧不共用参数，不直接复制数值，
  也不能将裸电机轴Kt直接当作Wuji关节侧Kt。
- 反射为 `tau_glove = -gain * J_retarget.T @ tau_wuji`，再经当前CAD的
  `J_motor^-T` 转换到模型电机力矩，按手套Kt和raw轴方向／尺度得到实际SDK输出。
- 完整重定向Jac是独立求解器的准静态局部导数；后台四个spawn进程每个计算3列，按同一请求号、原始姿态及源时戳组装完整20×12矩阵，不fork已经连接的SDK。
  待算输入仅保留最新一份，不与控制环争抢GIL。仍做精确CAD正负扰动、共同中心初值及同一官方优化目标；不修改正向Retarget算法。
  矩阵保留原始姿态时间及有效半径，不将计算完成时间伪装为源时间；真实反馈仍受100ms期限约束。
- 首次手套使能仍按原伺服规则回零，之后重新采样预热；力矩从0渐入。
  切回推理后手套恢复位置跟随，从实测模型m12平滑接入，同会话不重复回零。
- 软界处只抑制继续向外的力矩，允许回到界内；数值限幅历史以实际发出的电流为准。

校准JSON必需字段：

| 字段 | 含义／顺序 |
|---|---|
| `schema_version`, `calibrated`, `source_id` | 分别为1、真实标定确认true、可追溯标定来源 |
| `wuji_serial_number`, `glove_hand` | 绑定Wuji设备及手套left/right，启动前核对 |
| `wuji_torque_per_amp_nm`, `wuji_current_bias_a` | 20轴语义关节顺序的带符号非零关节侧等效Nm/A（含减速）、偏置A |
| `glove_torque_constant_nm_per_a` | raw电机ID1–12、与SDK原始角同一轴侧的正Kt，Nm/A；不是Wuji的常数 |
| `max_current_a`, `max_slew_a_s` | raw电机ID1–12各自的反馈电流上限A、斜率A/s，与手套驱动上限取更严格者 |
| `gain` | 反射缩放系数，0＜gain≤1，不是任一侧的kp或kd |

可选数值设置：`ramp_duration_s=0.5`、`jacobian_max_age_s=0.5`、
`jacobian_max_pose_delta_rad=0.25`、`feedback_max_age_s=0.1`、
`jacobian_refresh_interval_s=0.1`、`jacobian_step_rad=0.01`。这些是控制／数值设置，不是物理标定。

有真实标定后，令环境变量 `FEEDBACK_CALIBRATION` 指向该JSON，可使用：

```bash
pixi run -e hardware real-hand follow \
  --mode glove-to-hand2 --wuji-serial WH2KA01260814006 \
  --hand right --glove-hand right --approved-firmware 2.2.3 \
  --feedback-calibration "$FEEDBACK_CALIBRATION" --rate 50 --execute
```

不加 `--execute` 仅计算预览，双方均不使能／回零。缺文件、未标定、设备不匹配或缺实测effort时，
在输出前拒绝；本仓库不放假Kt标定文件，也不会把仿真系数写入真机配置。
`RealWujiControlLoop(..., force_feedback=controller)` 复用同一模块，接管电流只从实测effort计算，
不再接受任意 `haptic_current_a` 注入；自定义工厂必须可spawn序列化，可用 `RetargetPipelineFactory`。

当前现场手套Kt仍为null；Wuji20轴及手套12轴真实力矩标定、重力／摩擦补偿和带人手闭环尚未验收。
仅用电流偏置与Kt换算得到的是含内部负载的执行器反作用力矩估计，不能宣称为净接触力。

### 最大负载占比线性近似（不要求Kt标定）

按用户要求增加独立 `max_torque_linear_estimate` 模式。它使用实测电流比例作负载代理，
不声称Wuji已经测得Nm，也不把未知电机峰值或协议量程冒充真实最大力矩。
依据[官方控制指南](https://docs.wuji.tech/docs/en/wuji-hand/latest/control-guide.md)，
Wuji的effort是A、设备电流参数上限为2.0A；本模式以2.0A作为归一化参考，不写设备电流限值。

保留完整几何链路：`r=clip((I-bias)*sign/Imax,-1,1)`，
`B=J_motor^-T @ J_retarget.T`，按每行 `sum(abs(B))` 的最坏负载包络归一化，
再反向映射到手套各轴软件允许的名义Nm上限。单轴线性，联合负载仍有界；
四指未观测侧摆不参与反射，不建立手写12→20关节表。

现场右手参数文件：`config/feedback/max-torque-linear-right.json`。
`estimate_acknowledged=true` 表示明确使用近似，**不是** `calibrated=true`。
按用户明确选择，当前每raw轴 **10Nm软件名义上限**，低延迟配置斜率6Nm/s、会话启动0.5秒渐入；
`wuji-force-linear`与`wuji-force-contact`共用此限额。它不是固定输出10Nm，也不是已核定的电机允许值。
这些值不是电机额定/峰值或已验证安全力矩。偏置暂按0、方向按SDK正轴，需后续单指核对。
Wuji位置仍5/0.1，手套位置仍45/0.5，接管力控另用0/0；不混用两侧参数。

Encos的±10Nm只是12位协议编码范围，步长约0.004884Nm，零附近误差约±0.002442Nm。
驱动保留半LSB裕量，软界零附近选向内档；连续斜率历史不吸附编码网格，避免微小目标卡住。
名义上限提高不消除协议量化，也不保证真机连续线性的手感或名义0对应实测0。
无Kt时这里只限制名义Nm，**不宣称实际电流已由0.25A软件值限定**。
该限额不是厂商批准的最大值。要按官方电机上限配置，必须先匹配实际电机型号、
持续／峰值工作条件及SDK单位；不得把±10Nm编码范围或Wuji的2A参考当成手套额定能力。

局部Jac只用于其原始姿态附近：超过0.25rad或原始姿态年龄超过0.5秒时，
立即提交最新姿态，**不阻塞位置跟随，也不继续用旧矩阵反射力**。连续力矩目标直接归零
（仍存在上述协议零档偏差）。接触模式恢复后从零按既有斜率有界衔接，不重启会话渐入；线性／真实Kt模式保留原恢复渐入。不将旧矩阵或传感帧续期。
终端 `反馈状态=active` 表示当前投影有效，`refreshing_pose`／`refreshing_age`
表示等待姿态／年龄刷新，不代表力反馈正在生效。持续刷新说明当前运动／算力下不能连续反射力。
原始反馈过期、掉线、后台数值异常、当前CAD传动奇异或命令超时仍拒发并释放本会话输出。

首次真机检查先不佩戴手套；以下任务会使能两侧并先对手套伺服回零：

```bash
pixi run -e hardware wuji-force-linear -- --duration 10
```

通用入口为 `real-hand follow --mode glove-to-hand2 --feedback-linear-config PATH`；
与真实标定参数 `--feedback-calibration` 互斥。不加 `--execute` 仍只读两侧并计算；
普通 `wuji-teleop` 不变、不会自动使能手套力反馈。

### 受阻估计与虚拟弹簧阻尼反馈

仅新增力反馈，不修改既有Retargeting、坐标系、权重、骨架或正向位置目标。
`wuji-force-linear`仍为电流比例反射；新增`wuji-force-contact`显式选择接触阻抗估计：

```bash
pixi run -e hardware wuji-force-contact -- --curves-port 8765 --duration 10
```

**此任务会使能两侧并先对手套伺服回零；当前控制器尚未做真机或佩戴验收，切勿直接穿戴运行。**
通用入口同时使用`--feedback-linear-config`（设备绑定、偏置及独立名义输出限额）
和`--contact-feedback-config config/feedback/contact-impedance-right.json`（受阻/阻抗参数）。
手套驱动仍MIT `kp=kd=0`；Wuji仍位置MIT `5/0.1`。软件K/D不是SDK的kp/kd。

受阻判据为：最终Wuji目标与实测位置误差、电流负载及低实际速度同时满足，并持续驻留。
当前试验阈值为0.06rad／0.12A／0.2rad/s、驻留0.04秒；释放阈值0.03rad／0.06A，
迟滞0.06秒。缺原始速度时，需两次真实推进的反馈帧差分；不把缺速度当零速度。
重复源帧不推进驻留，大采样间隙重置；未观测侧摆不参与估计。
接触估计持续消费每周期新鲜状态，即使暂时等待有效几何也继续确认或释放；无效几何仅禁止实际力反射，不再清空受阻历史。此时`waiting_geometry`仍表示最终输出零，但诊断中的虚拟τ与接触掩码可非零，不能把它们误认为已施加的力。

参考Franka主手代码的连续负载权重和非线性速度阻尼，但Wuji只有电机电流，
没有同事代码所用的外部关节力矩输入，不能直接复制其力矩、重力补偿或增益。
对锁存受阻方向`n`，令`e=目标位置-实测位置`，`v=n*(目标速度-实测速度)`，
`I_f`为扣偏置后电流绝对值的低通；各轴计算：

```text
u = min(I_f / load_weight_scale_a, 20)
w = 1 - exp(-u²)
v = clip(v, -relative_velocity_limit_rad_s, +relative_velocity_limit_rad_s)
b = 0.6*sigmoid(u) + 0.4*sigmoid(min(abs(v)/stalled_speed_rad_s, 20))
tau_virtual = -n * max(w*K*max(n*e-error_off, 0) + b*D*v, 0)
```

`load_weight_scale_a=0.5A`是试验权重尺度，不是电流限额或标定值；
K=0.10名义Nm/rad、D=0.003名义Nm·s/rad均是软件虚拟系数。相对速度限幅2rad/s仅用于计算阻尼。
相对速度低于`-0.02rad/s`时立即清除该轴的虚拟反力及滤波历史；重复源帧中目标回开也只撤力、不推进驻留。
接触仍锁存时再停止回开，可从零恢复阻力；`contact`不保证本周期输出非零。
5Hz电流低通、8Hz虚拟输出低通后，复用原有`J_retarget.T`和当前CAD传动投影，
再乘接触配置的`reflection_gain`，不建立关节对应表、不额外重定向。
各手套电机增力和反向重新建立受斜率限制；减力、释放和几何失效立即生效，
避免其他手指仍受阻时保留已张开手指的旧反力。协议零档偏差仍存在。
原始反馈失效、数值异常、软限位和命令期限仍保持原约束。

终端及曲线显示`free / arming / contact / releasing / waiting_geometry`。
其中`contact`仅表示**受阻估计成立**，不能独立区分物体、摩擦或机构边界，
也不是接触力测量；实际电流比例、软件虚拟力矩与SDK名义指令分开报告。
当前每raw轴10Nm软件名义上限，接触模式`reflection_gain=10000`（此前6000，限幅前投影增加至约1.67倍）；
6Nm/s约束接触模式的增力，线性模式仍双向限速。首次start／显式reset保留500ms渐入，
运行期几何恢复后不重复该渐入。达到限额后不会再按倍率线性增加。
这不是硬件额定能力、安全力矩或实测手部力；上述阈值、权重及增益均需单指空载、
负载和实际电流／力矩标定后重新整定。配置在启动时加载，不支持热改。
此变更未上真机，不是穿戴安全验收；按官方上限配置还缺实际电机型号／铭牌，详见
[官方参数适用边界](docs/research/wujihand2-official.md)。

低延迟配置将几何提交间隔从100ms改为20ms，矩阵有效期0.5秒和姿态半径0.25rad保持不变。确定性消费者回归覆盖几何等待期间受阻/释放、恢复首20ms有界输出、首次渐入、旧源帧不续期及实际软限位。无硬件分片benchmark的五个姿态上，完整差分中位204.5ms、四进程分片72.8ms，矩阵最大绝对差为0；这不是电机响应或操作者手感测量。


### 双侧力控曲线与τ计算数值

在两个终端分别运行。**终端一只打开监视器，不连接或使能硬件：**

```bash
cd /home/boen/hand/wujihand2_force_glove
pixi run force-ui
```

**终端二会启动真机力反馈链路、使能两侧并先对手套伺服回零：**

```bash
cd /home/boen/hand/wujihand2_force_glove
pixi run -e hardware wuji-force-contact -- --curves-port 8765 --duration 10
```

`wuji-force-contact`默认精简终端输出：每秒打印一次J1–J12的SDK原始轴名义力矩前馈，
并标注提交状态和`active/contact`等反馈状态；这是SDK量化前目标，不是实测力矩。
首次只读预热及回零提示、SDK警告和错误仍会显示；曲线UDP数据不受精简输出影响。
`RUST_LOG=warn`只抑制SDK普通信息，不关闭警告；使用`pixi run -q`可隐藏Pixi自己的任务横幅。


窗口默认按食指打开“τ计算数值”页，可切换手指或全部通道：

- Wuji20语义关节表：最终目标跟踪误差rad、SDK实际速度rad/s、电流A、低通后的软件虚拟τ_v（未乘反射倍率）。
- 手套raw12电机表：几何投影并放大后的τ（渐入/限幅/限斜率之前）、最终SDK量化前名义前馈τ、SDK解码电流A。
- 放大系数来自控制端本周期，不读取UI本地配置猜测。三阶段τ不是同一坐标，不能逐项直接相减；前馈τ不包含完整MIT的位置/速度PD输出，也不是实测轴力矩或接触力。

新增诊断字段须重启数据源程序才会提供；旧包或缺失测量显示“未提供”，不补0。
“曲线”页保留Wuji电流/负载占比、手套最终名义指令及SDK解码电流三图；
名义Nm曲线以mN·m显示（Nm×1000），数值页直接用Nm并保留小数/科学计数。
真实标定电流模式明确显示A，不冒充Nm。关节分组只用于查看，不是一一关节映射。
两侧原始帧龄、是否发送目标和Jac刷新状态独立显示；active不是触觉验收通过。
暂停同时冻结曲线和数值到同一帧，但仍接收新包；清空或切换会话不让旧数值冒充当前。

控制完成发送后通过非阻塞本机UDP旁路发布，UI独立进程接收，绘图不进入控制循环。
曲线包允许丢失，不等待UI或重发硬件目标。端口自定义时同时使用
`force-ui -- --port PORT` 和控制端 `--curves-port PORT`。
**关闭窗口不会停止真机**；仍须在控制终端按Ctrl+C并确认设备停稳。
这不是旧的Encos标定GUI，不应同时用另一个程序占用手套通道。
仅查看界面可运行 `pixi run force-ui -- --demo`，所有曲线明确标为离线合成，
不连接硬件或网络，不代表物理仿真或接触力反馈效果。

### PlotJuggler ROS2实时控制链路监测

本机已将 ROS Humble 的 PlotJuggler 和 ROS 插件解包到用户目录
`~/.local/opt/plotjuggler-humble/root`，不是系统级apt注册安装。运行脚本使用系统
ROS Humble和Python，避免与Pixi的Python/Qt库混用；可用环境变量
`PLOTJUGGLER_HUMBLE_ROOT`指定相同布局的其他解包目录。

先后在三个终端运行：

```bash
# 终端一：只读UDP→ROS2旁路；同时将原包转发给既有 force-ui 的8765端口。
cd /home/boen/hand/wujihand2_force_glove
pixi run plotjuggler-bridge -- --forward-port 8765

# 终端二：只开PlotJuggler，不连接或使能硬件。
cd /home/boen/hand/wujihand2_force_glove
pixi run plotjuggler -- --nosplash

# 终端三：此命令会启动真实力反馈控制、使能硬件并先对手套伺服回零。
cd /home/boen/hand/wujihand2_force_glove
pixi run -e hardware wuji-force-contact -- --curves-port 8766
```

在PlotJuggler左侧 `Streaming` 选择 `ROS2 Topic Subscriber`，点 `Start`。
在选择窗口用过滤框筛选 `/glove_bridge/`，选中列表并按 `Ctrl+A`；**同时选中
`/names` 与 `/values` 话题**，勾选 `If present, use the timestamp in the field
[header.stamp]`，再点 `OK`。采集到数据后，展开左侧树，把相应曲线拖入图中；
相同关节的 `wuji_target_rad/.../position` 与 `wuji_position_rad/.../position`
可放在同一张图上比较。右手第7个语义轴为 `r_index_finger_pip`。
已知存在先订阅后才出现的轴时，需同时订阅其 `/names`，不能只选 `/values`。

| 数据 | ROS2话题前缀或变量 | 含义 |
| --- | --- | --- |
| 手套输入 | `glove_position_rad`、`glove_raw_position_deg_motor_N` | 12维真实指节rad、各电机原始角度deg |
| Wuji目标与反馈 | `wuji_target_rad`、`wuji_position_rad`、`wuji_current_a` | 已提交的位置目标、反馈位置rad、电流A |
| 手套力反馈 | `glove_nominal_feedforward_nm`或`glove_nominal_feedforward_a`、`glove_current_a_motor_N` | 手套名义前馈和SDK反馈电流；不是实测接触力 |
| 算法中间量 | `wuji_tracking_error_rad`、`virtual_hand_torque_nm`、`glove_projected_nominal_torque_nm` | 跟踪误差、软件虚拟反作用、映射后的名义力矩 |
| 跟随时移 | `wuji_following_lag_estimate_s`、`wuji_lag_estimate_valid` | 单轴目标到实测波形的时移估计及有效标记 |
| 旁路计时 | `glove_sample_to_wuji_submit_s`、`wuji_sample_to_glove_submit_s`、`control_cycle_duration_s` | 原始反馈接收到SDK调用返回或入队、单个控制周期耗时 |
| 帧龄与状态 | `wuji_frame_age_s`、`glove_frame_age_s`、`jacobian_reported_age_s`、`telemetry_age_at_bridge_s`、`stream_alive` | 接收时帧龄、Jac报告年龄、UDP到桥接处理时年龄、数据是否仍在到达 |

数值统计话题路径是 `/<变量>/values/<轴名>`，对应的 `/<变量>/names`
提供轴名；其中 `_s` 的单位为秒，`0.15`表示150ms。默认时移估计使用Wuji轴索引6；
可用桥接参数 `--lag-axis 0..19`指定其他轴。缺少足够运动、波形相关性不足、
时移超出0–0.5秒或数据中断时，有效标记为0且不发布新的时移值；无估计不代表零延迟。
该时移包含控制、通信和机械跟踪动态，**不是单独的通信延迟或力反馈端到端延迟**。
SDK返回/手套入队也不代表电机收到或物理输出力矩。手套q12由异步电机帧重建，
时间戳取最老帧；逐电机原始角度和电流使用各自接收时刻。
名义前馈、软件虚拟力矩和电流不能直接当作实测手部接触力。

只看界面可将第三个终端改为 `pixi run plotjuggler-demo`，此时是合成信号，
其中人为注入150ms的位置跟随时移，`telemetry_source_code`为0；真实硬件数据为1。
演示源与真机数据源不要同时运行。桥接器和图形界面仅旁路观察；丢包或关闭窗口
不会停止真实控制，仍需在真机控制终端按 `Ctrl+C` 并确认设备停稳。

## Encos 12轴电机手套接入

压缩包已部署到：

```text
hardware/encos_dexhand/
```

外层 Pixi 已统一提供：

```bash
pixi run encos-test       # 连接真实USB-CAN并持续打印12轴角度
pixi run gui              # 启动Encos PyQt5调试/标定界面
pixi run encos-zero-step  # 拇指到小拇指逐指按Enter记录软件零位
pixi run encos-home       # J1使用1.5A，其余轴1.0A按zero_deg自动回零
pixi run encos-live-viewer # 只读真实反馈，实时驱动MuJoCo手套和Hand2方向观察
```
`encos-zero-step` 不发送位置/MIT目标，只读取当前角度；每次回车确认一根手指后立即写入
`joint_calibration.json` 和 `glove_mapping.json` 的 `zero_deg`，不会固化硬件零位。

四指可独立标零，先退出遥操/GUI，确保电机失能：

```bash
pixi run encos-zero-index   # 食指J5/J6
pixi run encos-zero-middle  # 中指J7/J8
pixi run encos-zero-ring    # 无名指J9/J10
pixi run encos-zero-pinky   # 小拇指J11/J12
```

每次只处理指定手指；摆到伸直零位后按一次Enter，保存即退出，不要求其它手指处于零位。
以上四条任务同步更新本项目及`/home/boen/hand/dex_glove`的现场映射和SDK标定文件；
只修改所选手指，不复制其它轴配置、不修改PD和回零规则，也不发送使能或运动目标。
新零位同时平移原始角度软限位，使相对零位的屈伸范围保持不变。完成后重启遥操加载。
通用入口支持`--finger index|middle|ring|pinky|thumb`，可用`--sync-mapping PATH`
指定额外同步文件；不指定`--finger`仍按五指顺序逐指回车确认。

`encos-live-viewer` 不使能手套；读取 Encos 原始反馈后完成模型电机 FK，
直接使用同次求解的完整68维 CAD 树显示手套，再用真实指节经过共享
canonical 任务空间及 WujiRetarget 显示 Hand2。终端分列 `raw_deg`、
`motor_rad`、`joint_rad`，便于区分电机与指节、判断方向。
当前现场方向确认后的反向轴为 J4、J6、J8、J10、J12，已在
`hardware/encos_dexhand/configs/glove_mapping.json` 中设置 `"sign": -1.0`；
其余轴保持 `"sign": 1.0`。
限位均以保存的电机零位为基准：J1/J2相对范围为`[-90°,90°]`，J3–J12为`[0°,90°]`。
配置中的`min_deg/max_deg`保存换算后的编码器原始范围。J3/J5/J7/J9/J11为
`[zero_deg,zero_deg+90°]`；反向轴J4/J6/J8/J10/J12为`[zero_deg-90°,zero_deg]`；
J1/J2为`[zero_deg-90°,zero_deg+90°]`。零位和方向未改动，不得重复加零位。
这些零位、回零电流和软限位是硬件适配器的固有配置：后续位置控制与MIT触觉控制
都会经过同一套限位校验；回零入口固定读取同一文件中的 `zero_deg` 和分轴回零电流，
不需要在命令行重复传入。

`encos-home` 要求 ID 1–12 全部在线；先以当前反馈角度发送保持帧完成使能，
再向 `glove_mapping.json` 的 `zero_deg` 发送位置目标。默认判定为最大角度误差
不超过 `0.15 rad`；J1（ID 1）回零电流固定为 `1.5 A`，ID 2–12 固定为 `1.0 A`；
成功、超时或报警后都会自动失能。

```bash
pixi run encos-home -- --tolerance-rad 0.15 --speed-rpm 5 --timeout-s 45
```

算法侧 `EncosGloveHardwareDriver` 的位置接口始终接收真实指节 `GlovePose`，
不是原始电机角。完整 CAD 树贯穿读回、限位后的目标预览和任务 FK，
避免同周期反复求解同一个机构；输入采样与目标数值初值分别保存，不混用时戳。
实时控制固定 MIT `kp=45.0/kd=0.5`，零速度、默认零前馈；回零使用伺服。
本会话首次位置输出必须先 `home()`；`connect()`／只读 `prepare()` 不使能。
MIT 触觉前馈仍要求真实 `torque_constant_nm_per_a`，不使用仿真假设代替。
SDK 工作线程按原始命令期限停止重发过期 MIT 目标；反馈时戳来自实际接收，
不使用读缓存的时间。主机保护不能替代设备级断链保护或硬件急停。

`hardware/encos_dexhand/configs/glove_mapping.json`保存已确认的轴序、零位、符号、比例和
换算后的原始角度限位，当前`calibrated=true`，允许按这些边界发送MIT位置目标。
这不代表力矩常数已经标定；`torque_constant_nm_per_a`仍为空，触觉前馈命令继续拒绝。

## 快速开始

```bash
pixi install
pixi run demo
pixi run validate-assets
pixi run test
```

`demo` 只运行离线手套闭链和 WujiRetarget，不连接真实手套或 WujiHand2。`vendor/wuji-retargeting` 固定官方 `v2026.8.17` 源码版本；Wuji 设备 SDK 仍只允许通过硬件适配器接入。

### MuJoCo 力反馈仿真

```bash
pixi run mujoco-force-feedback
```

该任务加载官方 `wuji-description` 子模块中的 Hand2 Beta1 `right.xml`，使用官网
Control Guide 的 `kp=3.0`、`kd=0.05`、`effort_limit=1.5 A` 保守示例。仿真对食指指尖施加
1 N 外力，使用 MuJoCo 广义外力作为理想扰动观测，经过 `J_retarget.T`、vE3.3 闭链传动和
电流安全限幅，验证手套侧反馈信号和稳定性。
仿真使用 `feedback_jacobian_step_rad=0.01` 和 `feedback_gain=0.10` 抑制数值雅可比噪声；
这两个值同样不是官网真机参数。
此入口使用`CalibratedFeedback`的理想扰动力矩链路，**没有调用**
`wuji-force-contact`使用的`ContactImpedance`，不能用它的结果验收新的接触控制器。

仿真中的 A→N·m、手套电机力矩常数、人手保持刚度是显式假设，不是真机参数；仿真通过不代表
可以直接使能真机。

## 真实手套闭链模型

```python
from pathlib import Path
from wujihand2_force_glove.ve33_adapter import Ve33GloveModel

model = Ve33GloveModel("right", asset_root=Path("assets/hand_motor_ve33"))
glove_pose, motor_positions = model.solve(
    {name: 0.0 for name in model.kinematics.joint_names}
)
transmission = model.transmission()
J_motor = transmission.motor_jacobian(glove_pose.vector())
```

这里的 `motor_positions` 是物理电机轴角，`glove_pose.joints` 是12个真实指节坐标；两者不能互换。闭链模型的装配分支边界不是实物电机限位。

## 2026-11-03 离线验收与上电边界

`pixi run -e hardware test-glove-chain` 只运行手套机构、双向控制及相关驱动回归，
不运行全仓测试，不连接真机。已采集现场姿态经真实 Python 驱动、原 CAD
约束及官方重定向完成120帧离线回放、两次方向切换，注入25ms接收年龄：

| 离线方向 | 帧数 | 整周期最大值（ms） | 最终反馈年龄最大值（ms） |
| --- | ---: | ---: | ---: |
| Hand2 → Glove（两段） | 80 | 29.756 | 54.714 |
| Glove → Hand2 | 40 | 6.996 | 31.946 |

回放检查实际输出 FK、硬件软件限位和100ms原始期限，不发送任何目标。
MuJoCo 1800步反馈闭环稳定并检测到反馈效应；Kt、人手刚度和理想外力观测仍为仿真假设。
2026-11-03用户提供两方向各5秒、20Hz真实只读日志，未报告超时或无解；仅短时只读通过。
随后实控曾提交多帧目标，但多轴上报`Enc1BitRate`，再发生Zenoh无应答；
失能及急停请求也未获回包，进程退出不能当作设备已失能。先实体急停或断电确认停止，
再排查通信与编码器状态；连续实控、姿态行程与接管运动仍未验收通过。
真实触觉仍缺 Kt 与电流—力矩标定，禁止使用仿真常数解锁前馈。


## 目录

- `src/wujihand2_force_glove/`：领域数据、vE3.3任务点源、官方重定向适配、反馈投影、限幅、闭链适配器和硬件接口。
- `src/wujihand2_force_glove/real_hand/`：独立并列真机驱动、原生安全核心、双向运行入口。
- `hardware/encos_dexhand/`：部署后的Encos 12轴SDK、CAN工作线程、PyQt调试界面、配置和示例。
- `hardware/encos_dexhand/configs/glove_mapping.json`：现场已确认的电机轴序、零位、方向与原始软限位。
- `config/retargeting/official/`：左右手 Hand2 WujiRetarget 配置，URDF路径指向本项目资产。
- `vendor/wuji-retargeting/`：固定版本的官方重定向源码。
- `assets/hand_motor_ve33/`：用户提供的手套左右手模型与闭链配置。
- `assets/wuji-hand2/`：用户提供的 WujiHand2 左右手 URDF 和网格。
- `docs/PLAN.md`：里程碑、硬件未到时任务和首次真机禁止项。
- `docs/research/wujihand2-official.md`：官方一手资料研究与引用。

## 明确不做的事

- 不按 SDK 数组位置猜测模型轴序；真机必须用 `nid` 和逐轴小步实验确认。
- 不把电流 A 直接当作 N·m 或指尖力。
- 不在没有 watchdog、急停演练、状态新鲜度检查和限位标定时使能。
- 不声称已经实现完整20自由度或完整力反馈。
