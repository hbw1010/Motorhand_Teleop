# Wuji / Sharpa 并列控制实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以并列设备模块交付Sharpa仿真遥操、真机遥操力反馈、录制回放和人手接管，保留Wuji现有行为。

**Architecture:** C++共同控制层负责几何工作器、会话模式、原始期限和手套输出；Wuji/Sharpa并列实现设备、模型、重定向及带明确单位的反馈策略。20/22维由编译期参数或等价的严格维数契约表达；Python只承担入口和只读显示，不另写一套在线控制流程。

**Tech Stack:** C++17、Eigen、Pinocchio、NLopt、MuJoCo 4.0.0、固定Wuji C SDK、Sharpa Wave SDK 5.0.11、项目Pixi/Python 3.12。

**Spec:** `Motorhand_FFB/docs/superpowers/specs/2026-11-16-sharpa-parallel-control-design.md`

## 全局约束

- 用户已确认Wuji/Sharpa并列，设备特异逻辑分别实现；Sharpa首版只用SDK关节力矩，触觉关闭。
- 原始命令/反馈期限100ms；重复缓存不续期；只读路径不得使能、清错、回零或提交目标。
- 不修改用户最新手套标定、12轴10 Nm配置、G4时间戳及仿真回撤修复。
- 不复制现有会话、接管和手套输出保护形成两份实现；不把Sharpa逻辑塞进Wuji类，不截断22轴。
- 保持现有Wuji命令、配置、C ABI和轨迹兼容。允许薄兼容包装，禁止保留两份运行中的核心状态机。
- 新在线求解、反馈、MuJoCo推进和硬件提交使用C++；不引入ROS，不导入相邻工程运行时模块。
- Sharpa SDK完整本地安装且不入Git；设备SDK/模型缺失时具体报错，不伪造成功或反馈。
- 本任务不实现尚未落地的Franka式force_reflection或动力学辨识，不接入触觉。
- 真机开发验证先限于明确的只读操作；离线测试只用假设备或伪终端，不自动使能或移动用户设备。

## 重点审阅

1. 20轴旧数据误用到22轴：描述符和轨迹校验必须拒绝，任务1/5覆盖。
2. Sharpa设置模式会隐式使能：只读连接不调用，正式使能先装载实测保持目标，任务3覆盖。
3. 收包时间system_clock→monotonic：重复/乱序/时钟跳变不能刷新原始期限，任务3覆盖。
4. 力反馈单位和符号：A与SDK力矩分支独立，仿真环境力矩转换方向仅一次，任务3/4覆盖。
5. 接管抓握保持和原始目标过期：切换只允许一个手套输出主源；不借新状态续期旧推理输入，任务1/5覆盖。

## Task 1: 建立共同控制契约并迁移Wuji为并列模块

**文件范围：** `Motorhand_FFB/src/motorhand_teleop/real_hand/native_control/common/`、`hands/wuji/`；既有`types.hpp`、`session.*`、`inverse_*`、`force_*`、`jacobian_*`、`glove_output.*`、`trajectory.*`、`CMakeLists.txt`及对应原生测试。

**接口：**
- 创建共同的 `JointArray<N>`、`HandState<N>`、`HandDescriptor<N>`、`HandLoadUnit`、`RetargetJacobian<N>`；状态含真实N维位置/速度/负载、可用性、原始时间、状态和身份，Jacobian为N×12。
- `HandModel<N>`提供`solve(Points)`、`hand_points(JointArray<N>)`、`task_scales()`、`seed()`、`reset()`及独立求解器创建能力。
- `HandDevice<N>`提供只读`connect/read`、显式`enable/home/send(original_deadline_ns)/disable/close`及身份/限位描述。
- `HandFeedback<N>`拥有设备负载判据与反馈策略，产生共同手套规划所需的接触mask/方向/虚拟力矩/投影信息；共同层不将负载单位强制解释成A。
- `ControlSession<N>`复用既有切换、异步逆解/Jacobian和唯一手套输出路径；原`Session`保留为Wuji20的兼容入口。需要新增虚拟接口的方法由实际调用需求决定，不添加未使用的接口。

- [ ] 为共同状态的20/22维、单位身份校验、期限和切换加入失败测试；用现有Wuji原生测试固定回归。
- [ ] 将Wuji模型、SDK身份/设备及电流反馈策略迁入`hands/wuji`；共同的几何、会话与输出算法参数化为N。旧头文件可薄转发，兼容字段在入口转换。
- [ ] 保持Wuji优化参数、零位对齐、真实侧摆FK、跟随/接管及输出保护不变；构建不要求Sharpa SDK。
- [ ] 运行`pixi run test-control-native`和Wuji相关Python接口测试；对22维假设备运行同一共同会话的跟随/切换测试。
- [ ] 提交这一可独立验证的迁移。

## Task 2: Sharpa模型与22轴重定向

**文件范围：** `native_control/hands/sharpa/model.*`、`retarget.*`、设备描述；`Motorhand_FFB/config/retargeting/sharpa/`、用户模型资产、`tests/test_sharpa_model_native.cpp`及构建接入。

**消费：** 任务1的`HandModel<22>`、共同21点、vE3.3闭链输入。

**产生：** `hands::sharpa::Model`，左右侧22命名关节、URDF与MJCF/ctrl限位交集、FK任务点及独立可复制的重定向求解器。

- [ ] 写左右模型22维、关节/执行器对应、非法手侧、限位、固定四指侧摆、退化输入和重复调用连续性测试。
- [ ] 实现C++模型和重定向：使用Pinocchio/现有优化依赖；按掌局部坐标、骨段方向与指尖任务求解，不直接复制电机角。Sharpa额外自由度的固定/正规项策略写入独立配置。
- [ ] 验证同一手套轨迹可生成22维有界目标；正逆任务点一致、差分Jacobian与解析几何符号正确，不修改Wuji模型或参数。
- [ ] 保留用户资产许可/来源，校验引用文件；记录此处数值实现的本地来源与差异，不导入其他工程。
- [ ] 运行新模型测试及既有Wuji重定向/逆解回归，提交。

## Task 3: Sharpa SDK设备与关节力矩反馈

**文件范围：** `native_control/hands/sharpa/device.*`、`identity.*`、`feedback.*`、`transport.*`；`tools/install_sharpa_sdk.py`；`config/feedback/sharpa-*.json`；原生SDK替身/反馈测试及构建接入。

**消费：** `HandDevice<22>`、`HandFeedback<22>`、`hands::sharpa::Model`、官方5.0.11 SDK。

**产生：** 不依赖Wuji SDK的Sharpa真机模块；SDK报告关节力矩来源的接触反馈和负载归一化反馈。SN/PN/固件由配置显式绑定，启动只读发现可填写现场身份。

- [ ] 建立SDK运输接口替身，写只读无输出、身份不符、22维缺失/NaN、重复/乱序/过期、时钟跳变、已被外部使能、模式隐式使能、超时失能和重复关闭测试。
- [ ] 安装脚本从用户归档核验固定SHA256/版本，完整保留许可于`Motorhand_FFB/.cache/hardware-sdk/sharpa/5.0.11`；CMake可选开启Sharpa硬件，链接`libsharpa-wave-sdk.so`。
- [ ] 实现官方C++接入，关闭触觉和自动时间同步；保存SDK原始序号/时间，保守转换主机system_clock收包时间，不用轮询时刻续期。
- [ ] 使用独立的Sharpa力矩偏置/符号/阈值/负载参考配置，默认标记SDK报告值而非净外力。不可直接套用Wuji A阈值或伪造Kt。
- [ ] 真机同一SDK状态中的力矩缺失时停止反馈并明确诊断；位置遥操能力与反馈能力分别校验。正式使能按用户指定的`/home/boen/hand/dex_glove/real_hand/hands/sharpa_hand.py`参考，由SDK控制源切换同步实际关节到发送器状态，不先发送裁剪后的放松姿态，不在只读阶段设控制模式。
- [ ] 验证关节力矩→22×12几何投影→12轴手套输出的符号、耦合、限额、释放；执行SDK替身与共同控制回归，提交。

## Task 4: MuJoCo机械手后端与仿真遥操

**文件范围：** `native_control/simulation/`、共同假手套/模拟设备、Sharpa仿真CLI、`tests/test_sharpa_simulation_native.cpp`及Pixi任务。

**消费：** `ControlSession<22>`、Sharpa模型/反馈策略、MuJoCo单侧资产。

**产生：** 假手套→Sharpa纯离线仿真，以及真实Encos手套→Sharpa虚拟物体交互；共用同一Sharpa控制和接管逻辑，区别仅为设备后端。

- [ ] 写无硬件离线运行、目标只写ctrl、动力学跟随滞后、接触力方向、受阻反馈和释放、模式切换测试。
- [ ] C++加载MuJoCo并按0.002s步进；以原始仿真状态生成有源时间的22轴状态，接触力通过同姿态接触点Jacobian转换关节负载，不用驱动电流常数冒充仿真力。
- [ ] 提供可复现的固定物体抓握场景和只读快照，Python显示消费快照；假手套模式不加载真实SDK/串口，真手套输出需显式启用。
- [ ] 加入`sharpa-sim`与`sharpa-sim-teleop`入口；运行无窗口抓握/释放和推理→接管场景，保存验收指标并提交。

## Task 5: 对等CLI、轨迹、监视器与整体接管验收

**文件范围：** 共同/设备CLI、`native_control/trajectory.*`和`telemetry.*`兼容入口、`teleop_monitor/`、`pixi.toml`、README/ARCHITECTURE/CONTEXT和相关测试。

**消费：** 两个已接入的设备模块、共同会话及模拟后端。

**产生：** Sharpa发现、遥操、接触反馈、线性负载反馈、录制、回放及接管命令；显示与文件均明确22轴、型号和负载单位。保留所有旧Wuji入口。

- [ ] 为22维轨迹往返、旧Wuji轨迹读取、跨型号/手侧/关节顺序拒绝、UI22关节与单位、推理输入过期和抓握接管连续性写失败测试。
- [ ] 实现设备创建时选择；公共循环只持有接口，不在Wuji类中判断Sharpa。轨迹包含设备描述符和单位，旧schema按Wuji兼容读取。
- [ ] 增加`sharpa-discover`、`sharpa-teleop`、`sharpa-force-contact`、`sharpa-force-linear`、`sharpa-record`、`sharpa-replay`、`sharpa-takeover`及查看命令；默认参数不自动取得实体输出权。
- [ ] 使用共同模拟场景验证完整“手套跟随→机械手受阻→反馈→回放→人手接管→释放”，包括实际执行目标和手套模型电机角的连续性。
- [ ] 运行全部Python/原生构建与测试、Sharpa两侧离线烟测、只读设备身份/反馈预检。更新文档、记录未做的硬件运动验证并提交。

## 完成条件

任务1–5全部完成且最终独立审阅通过。提交记录和验证结果写入本计划专属工作记录；原`sim2.0`基线保留到整体验证完成。用户未要求本轮立即推送，完成后报告本地版本及可运行命令。
