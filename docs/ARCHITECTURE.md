# 项目架构

```text
GloveDriver
    └─> vE3.3 闭链求解器
        └─> 21 个 MediaPipe 顺序任务点
            └─> 官方 WujiRetarget
                └─> Hand2 URDF 语义关节顺序
                    └─> WujiTarget ──> WujiDriver

WujiState ──> 电流/力矩标定 ──> 重定向链路 Jᵀ
           ──> 手套独立坐标力矩 ──> 传动雅可比 ──> 电机电流 ──> GloveDriver
```

## 外部 seam

- `GloveDriver`：读手套12个独立角、闭链电机反馈；发送位置/力位混合命令。
- `WujiDriver`：按序列号连接、读取完整状态和诊断；发送恰好20项目标；执行 enable/disable/emergency-stop。
- 两个 driver 都是 Protocol，不在控制核心内创建 SDK 对象。

## 深模块

### `Ve33LandmarkSource`

调用 vE3.3 原始闭链模型，把12个独立输入求解为21个任务点，并执行与 `dex_glove` 一致的 thumb CMC 中点处理。它不把手套12维硬编码为 Hand2 20维。

共享 `ve33_geometry.py`／`native/ve33_geometry.cpp` 编译原资产的 URDF FK、
闭链点残差及解析雅可比。`Ve33Kinematics` 保持原 Newton／边界／分支检查；
逆解仍用原目标、正规项和真实电机约束，不新增第二套映射或放宽精度。
Python 原资产求解器保留作元数据来源与独立差分基准，不是实时回退路径。
几何核不依赖 Wuji SDK，与 `real_hand/native` 的硬件安全核分开安装和维护。


### `WujiHand2Retargeter`

调用官方 `wuji-retargeting` 优化器。vE3.3 点源先经过统一 canonical 掌坐标，再直接调用官方 Hand2 优化器；输出按 Hand2 URDF 的20个语义关节名重排。当前四指 MCP 侧摆通过 `retarget.fixed_joints` 锁为零并参与优化器边界约束，不再从21点几何推断。

### `BidirectionalRetargeter`

正向和逆向共享同一 `HandTaskState` canonical 21点任务空间。正向路径为
`GloveTaskModel -> WujiHand2Retargeter`；逆向路径为 `Hand2TaskModel -> GloveInverseSolver`。
两条路径可以同时计算候选目标，但 `ControlMode` 只允许当前主导方发送位置命令；控制权切换由
`HandoverController` 从当前命令连续过渡，避免重新初始化求解器造成位置跳变。

### `GloveWujiRetargetPipeline`

组合任务点源和官方重定向器。正常接管路径只执行一次正向重定向；力反馈需要的
`J_retarget = ∂q_wuji/∂q_glove` 通过独立求解器低频中心差分估计，禁止使用手写12→20映射代替。

### `CalibratedFeedback`

调用者提供 Wuji 的 A 值和完成标定的参数，模块负责偏置、每安培等效力矩和完整重定向雅可比的转置投影；缺少标定或雅可比时拒绝生成触觉输出。

### `SafetyLimiter`

调用者提供候选电流，模块负责有限值、每通道限幅、斜率限制、命令期限和故障锁存。它不能替代 Wuji 或手套的硬件急停。

### `WujiHand2HardwareDriver`

实现位于 `real_hand/wuji_hand2.py`，使用同目录私有 ctypes 绑定和 `native/` C ABI，
不导入 DexGlove 或 `HandPose`。官方 SDK 固定版本并本地构建，连接按显式 SN、
手侧和实测固件核对，批准版本不冒充实测版本。只读连接／关闭不取得输出权。
ABI6状态为20维 `WujiState`，同一joint_states包提供位置、真实速度rad/s及effort电流A，
按NID重排并保留原始收包时刻；可选量无效返回None。逐帧命令仍只发位置，非零速度/effort拒绝。

### `RealWujiControlLoop`

实现位于 `real_hand/control.py`。推理输入 q20 同时驱动 Hand2 和 Glove；
手套位置从最终已平滑的 Hand2 目标逆解，不从未平滑候选反解。
Hand2 保持已发命令锚点，Glove 独立从实测模型电机 m12 五次平滑接入。
成功逆解携带本次完整树，经 `position_from_tree` 取可发 m 目标，
`position_from_motors` 做限位与真实 FK，输出仍是真指节 q12，不再次 q→m 求解。
接管取消Glove位置跟随；配置真实力反馈控制器后，手套保持使能但只发MIT纯力矩，不持位。
双方输入、求解和预检均在任一侧发送前完成；实测越软件界时限幅优先于连续性。

### `EncosGloveHardwareDriver`

把部署在 `hardware/encos_dexhand` 的 Encos SDK 收敛到 `GloveDriver` seam：
SDK 原始 degree → 模型 m12 rad → 机构 FK → 真 q12；
目标真 q12 → 机构 IK → raw degree 软限位 → 必要时 FK，触边不是停机条件。
同次限位 FK 的完整68维树通过 `position_tree(pose)` 传给共享任务模型，
`state_from_tree` 全校验后直接求点，不把实际输出重新从零逆解一次。
反馈 seed 与目标 seed 只用于数值初值，均不替代真实采样或修改模型已提交姿态。
现场位置映射已标定；缺轴序／方向／零位／限位拒发。标定电流模式缺Kt拒发，显式名义Nm近似模式另行配置。
首次 Motor 输出先伺服回零（5rpm，J1=1.5A，其余1A，0.15rad），正常 MIT 45/0.5。
位置跟随45/0.5与接管力控0/0是独立模式；Kt按raw ID分别标定，模型A→raw A→SDK Nm只换算一次。
在实际m边界只截断外向电流，返回实际电流供上层同步斜率历史；零位和raw范围不被传感值裁剪掩盖。
SDK 按真实接收时刻拒绝旧数据，并在 CAN 工作线程检查命令期限。

### `RealWujiControlLoop` 的双端设备模式

传入 `glove_driver` 后，推理模式反解最终已平滑Hand2目标，驱动手套位置跟随。
传入 `ForceFeedbackController` 后，接管以手套实测q12控制Wuji，同时把Wuji实测effort转换为手套阻力。
不再从外部任意 `haptic_current_a` 注入接管电流；两侧完整预检先于任一发送，异常释放已拥有输出。

### `ForceFeedbackController`

实现位于 `real_hand/force_feedback.py`。严格加载版本化单机标定并绑定序列号／手套侧，
Wuji20轴Nm/A、手套12轴Kt与各自输出限制独立，不与kp/kd混用。
负号反射执行器力矩，经完整重定向Jac转置与当前实测树的CAD隐函数Jac，生成模型电机电流。
spawn子进程独占同配置求解器；管理线程只做IPC、最新待算输入容量1，实时计算不等待完整差分。
数值求导复用独立优化器及CAD闭环种子；每次正负扰动清空历史正则，中心初值仅用于数值加速，
独立求导器收紧精度，不改正常位置求解器配置。它是准静态局部导数，不是优化历史的动态导数。
缺真实标定、effort、初始化矩阵或当前闭链Jac奇异均拒绝；幅值／斜率限幅、五次渐入与实际发送确认独立管理。
正常运动可快于低频差分：局部矩阵超姿态半径／年龄时强制提交最新姿态，
返回显式refreshing_pose／refreshing_age，连续力输出立即归零，不用斜率历史保留旧姿态力。
位置跟随不等待worker；新矩阵覆盖当前姿态后按模式的增力斜率有界恢复并返回active。矩阵原始时戳不续期，
未来时戳、真实反馈失效、后台错误及传动奇异仍属故障，不走等待分支。
该模块不连接设备、不使能、不发送；硬件所有权留在Runtime/Core。

`MaxTorqueLinearConfig` 是独立估计配置，不带calibrated字段。实测A只生成 `wuji_load_fraction`，
经完整B矩阵及逐行L1包络得到有界名义Nm；估计模式的Wuji/指节绝对Nm字段为None，不能假造量纲。
两种配置复用同一Controller与几何进程池，公共参数入口为 `config`，不将近似配置称作标定。
手套显式Nm模式直接发送0/0 MIT前馈，Current/Kt接口与Nm接口互斥；归一化参考不改Wuji设备电流参数。
协议半LSB裕量与软界向内零档由硬件seam处理，SDK参数确认与连续斜率历史区分，避免量化偏置积累。
名义输出不等于实测力矩；当前按用户选择设为每raw轴10Nm软件上限，不证明设备额定能力或实际电流限制。

`contact_feedback.ContactImpedance`只读最终Wuji目标、原始电流和位置/速度，
在源时间上实现受阻驻留/迟滞、滤波、基于电流的连续负载权重和相对速度非线性阻尼；
参考Franka主手控制律，但不把Wuji电流伪装为外力矩，也不调用或修改Retarget。
相对速度限幅；张开超过速度死区时立即清除该Wuji轴的虚拟反力，锁存状态仍可维持。
重复源帧目标回开只能清旧力，不能推进驻留、滤波或速度差分。
`ForceFeedbackController(..., contact_config=...)`仅在明确的名义Nm限额模式启用它，
计算`J_retarget.T @ virtual_hand_torque_nm`后复用原CAD传动，再乘接触专用`reflection_gain`。
手套各模型电机增力受斜率限制；减力立即生效，换向先归零再按斜率建立，避免多指接触时残留旧力。
放大发生在原raw轴限幅、渐入和边界投影之前；虚拟Nm及名义输出均不是实测力矩。
接触候选/释放/等待几何分别报告；几何失效不保留旧力矩，协议零档误差仍存在。



### `PairRuntime`

实现位于 `real_hand/runtime.py`。默认glove-to-hand2仅Wuji位置输出，Motor只读；
显式真实标定或已确认的名义线性配置启用反馈时，Wuji位置输出＋Motor纯力矩输出，仍只有手套一个位置主源。
hand2-to-glove仅Motor位置输出，Wuji只读不发hold；connect/inspect/prepare不取得输出权。
输出需显式 `execute`；切换先释放旧输出，新跟随端从实际反馈锚定，同会话只首次
Motor 输出回零。异常／关闭仅停止本会话拥有的输出，不影响只读输入端的所有者。
每周期保留100ms原始期限；慢求解、过期／未来反馈、无解、位置映射未标定或反馈配置缺失仍拒发。
这不同于外部推理 q20 同时驱动两侧的 `RealWujiControlLoop`，不能混为同一入口。


### 只读曲线旁路

`PairTick.glove_state`保留同周期原始手套反馈；`curve_telemetry.CurvePublisher`
在CLI完成双方发送后读取这个帧和已确认的输出，不额外访问设备。模型轴目标只按
已标定的sign×scale换算到raw ID轴，Nm与等效A分开标识；缺失测量保留None。
`--curves-port`显式启用非阻塞本机UDP，未开启时不创建传输对象；曲线丢包不阻塞、
不重发控制，不改变既有deadline／回零／增益。独立`force_curves`进程只收数据，
不连接SDK或取得输出权；`force_curves_view`用QPainter显示有界历史并标识帧龄／暂停／失效。
关闭监视器与停止控制是两件事，界面不得将历史发送状态称作当前使能确认。

## 数据契约

- 位置：rad；任务点：m；Wuji effort/current：A；时间：本机 monotonic ns。
- Wuji 状态必须按 `nid` 归一化为项目定义的20个关节顺序；少于20个在线关节时进入降级，不发送旧目标。
- `lateral_policy=fixed_zero` 表示四指 MCP 侧摆被硬约束为0；只有恢复可观测侧摆输入后才能改回几何推断策略。
- 标定模式需真实 `calibrated=True` 与可追溯参数；名义线性模式需显式 `estimate_acknowledged=True`，不能伪装标定、实测Nm或净接触力。
