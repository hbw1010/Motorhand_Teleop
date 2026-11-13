# 复用手套抓虚拟物体反馈链路的接入核对

用户希望复用现有仿真的driver_impedance及摩擦补偿。现已完成原生接入，`force-contact` 默认使用驱动阻抗及有效自由空间摩擦补偿，保留 `--feedback-mode torque` 回退。完成离线测试，未进行穿戴真机验收；具体配置和命令见 `../../config/feedback/driver-impedance-native.md`。

源为 `sim/config/default.yaml` 与 `sim/src/glove_sim/impedance.py`。阻抗使用接触代理角、MIT kp/kd，摩擦补偿只在自由空间输出：普通轴Fc=0.25Nm、J2为0.10Nm，速度死区0.02rad/s、tanh尺度0.08rad/s、速度低通12Hz，补偿上限0.6Nm。现有Wuji纯力矩入口不能直接读取这组driver kp/kd作为20维虚拟阻抗。

接入时采用的边界：

- 新增12轴阻抗规划及Encos写入接口，沿用单写线程、原始100ms期限及故障撤力。
- 从Wuji目标误差、电流及速度估计受阻代理，不冒充真实接触面；可使用最终有效Jacobian与传动映射构造局部有界代理角。
- 使用同一零位/轴向。若逆解Wuji姿态，需要处理当前zero_reference；不能直接将零位对齐后的Wuji角用于旧绝对逆解。
- raw/model换算：令a=sign×scale，则目标raw角=zero+model/a，kp_raw=a²kp_model，kd_raw=a²kd_model，tau_raw=a×tau_model。
- 摩擦补偿只对有效自由空间启用；缺帧、Jacobian无效、模式切换及停止均清零，不把故障当成自由空间。
- 源margin只限制弹簧项，并不能约束阻尼叠加后的总PD。离线复现：kp60、kd0.48、dq2时，弹簧-3.4Nm但总PD为-4.36Nm；慢速撤离也可因阻尼得到轻微回拉。因此移植需总名义预算、量化余量及单侧撤力约束；不能宣称MIT协议具有硬力矩限额。

真实Wuji没有仿真物体硬度来源，需要明确固定刚度/硬度设置；初版Wuji保留4/2Nm上限，没有自动采用仿真J1的6Nm；2026-11-13用户先要求J1/J2各5Nm，随后调至各10Nm，其余各4Nm。当前拇指+20%的调整属于旧软件虚拟阻抗，不应与未来driver kp/kd重复放大。

新UI已显示实际提交的摩擦前馈分量和阻抗参数；旧模式没有这一分量时保持未提供。
