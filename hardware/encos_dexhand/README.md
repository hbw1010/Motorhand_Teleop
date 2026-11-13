# Encos 12-Axis Dexterous Hand SDK (v0.3.0)

## 1. 快速上手 (Pixi 工作流)
pixi run test   # 终端自检 12 轴并实时打印物理角度
pixi run gui    # 启动图形化调试上位机

## 2. 算法端 Python SDK 调用示例
from encos_hand import EncosHand

hand = EncosHand()
hand.connect()
print("在线电机:", hand.scan())

# 实时获取全手 12 轴角度
print("当前角度:", hand.get_all_angles())

# 伺服位置控制
hand.set_position(motor_id=8, target_deg=35.0, speed_rpm=30.0)

# 运行时固定参数：kp=45.0、kd=0.5；正常适配器控制使用MIT力位混控。
hand.set_mit(motor_id=8, target_deg=20.0, kp=45.0, kd=0.5)

hand.close()

## 3. 外层项目算法接入

外层项目通过 `wujihand2_force_glove.EncosGloveHardwareDriver` 使用本 SDK：

```python
from wujihand2_force_glove import EncosGloveHardwareDriver

glove = EncosGloveHardwareDriver()
glove.connect()
glove.prepare()          # 只读预热实际CAD装配；该冷帧不充作控制输入
pose = glove.read_pose()  # 新鲜GlovePose：真实指节rad，不是SDK原始电机角
```

电机 ID 到模型电机轴的绑定、原始角零位／方向及软限位配置在：

```text
configs/sim/glove_mapping.json
configs/ffb/glove_mapping.json
```

两套标定分别供外层仿真和 FFB 使用，不可未经设备核对互换。各目录中的 `joint_calibration.json` 属于对应标定。

当前已确认的电机 ID 顺序为：

```text
1–4   拇指：指根 → 指尖
5–6   食指：指根 → 指尖
7–8   中指：指根 → 指尖
9–10  无名指：指根 → 指尖
11–12 小拇指：指根 → 指尖
```

现场轴序、零位及方向已确认。J1/J2相对零位范围为`[-90°,90°]`，J3–J12为`[0°,90°]`；
`glove_mapping.json`的`min_deg/max_deg`已换算成原始编码器范围，`calibrated=true`。
正向轴以`zero_deg+相对角度`换算；反向轴J4/J6/J8/J10/J12的原始范围为`[zero_deg-90°,zero_deg]`。
力矩常数仍未标定，不能把位置控制许可当作已标定的电流→Nm换算许可。
外层新增显式名义Nm近似模式时，直接调用SDK torque_nm，不填假Kt或宣称真实接触力。

## 4. 原始反馈与命令期限

`get_all_angles()` 返回编码器原始 degree；`get_all_feedbacks()` 返回同次接收的
`angle_deg`、诊断及 `received_ns`（主机 monotonic ns）。通道更新和快照读取加锁，
不能拿当前读缓存时间替换实际接收时刻。模型电机角和人体指节角的非线性机构转换
由外层 `real_hand/motor_hand.py` 完成，不能把减零位的电机角直接当关节角。

`set_mit(..., deadline_ns=...)` 与 `set_position(..., deadline_ns=...)` 接受可选原始期限；
外层控制器始终传入。过期后 CAN 工作线程不再重发旧驱动目标，转发 0A 状态探针。
不传期限的旧 SDK／GUI 调用维持原接口，调用者必须自行管理停止。
这是主机侧保护，不等价于已验证设备断链保护或硬件急停。

位置跟随MIT固定45/0.5；接管纯力矩前馈另用0/0。首次使能仍由外层servo回零，J1=1.5A、其余1A、5rpm、0.15rad。
前馈 `current_limit_a` 检查不代表可以观测或限制 MIT PD 产生的全部实际电流；
无真实 Kt 及电流／力矩标定时，不得把零前馈位置跟随宣称为真实接触力反馈。

MIT的±10Nm是12位编码范围而非电机能力，步长约0.004884Nm；名义0也存在半LSB量化误差。
显式Nm近似模式以外层软件限额为准，保留编码裕量并在软界选择向内零档；
它不要求Kt，但仅是SDK名义输出，不能把名义Nm限额解释为实际电流或实测力矩保证。
