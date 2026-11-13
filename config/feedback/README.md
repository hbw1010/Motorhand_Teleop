# Wuji 右手反馈配置

2026-11-12 按用户要求，将仓库 `tianji_teleop/config/motorhand_feedback/` 两份右手反馈 JSON 的参数迁移为当前 Wuji 配置，统一使用 `wuji-` 文件名与 `source_id`。初次迁移的参数与 `Motorhand_FFB/tianji_teleop/` 中的源副本相同；后续现场调整记录如下。

- `wuji-max-torque-linear-right.json`：J1、J2上限各10 Nm，J3–J12上限各4 Nm，增力斜率60 Nm/s，Wuji电流负载归一化参考1.5 A，几何刷新间隔0.05 s。
- `wuji-contact-impedance-right.json`：反馈倍率1500、电流滤波10 Hz、输出滤波25 Hz、接触确认0.03 s、释放确认0.15 s，其余参数均按源文件。

当前设备绑定保留为 `WH2KA01260813020`，不沿用源文件的另一台设备身份。手套零位、方向、比例和现场限位未更改。旧文件名为指向上述文件的相对符号链接，避免已有脚本引用失效或出现两套参数。

原生Wuji入口、Python推理/回放默认值与仿真配置中的反馈路径均指向新文件。这里只迁移反馈JSON；Tianji ROS2节点YAML中的200Hz、CANable时隙、跟随刚度、回零及对指设置不由这两份JSON控制，未移植其ROS2运行器。当前命令行 `--rate 50` 仍为50Hz，G4网关继续使用既有驱动。

1.5 A是此配置中电流负载归一化参考，不代表本次写入了Wuji设备的电流限制。软件名义Nm上限与实测力矩不同。原配置已备份到工作区 `.calibration_backup/wuji-feedback-*`，需要重启控制进程才会加载新值。

## 拇指反馈微调

2026-11-12 根据用户手感反馈，仅将WuJi拇指四个关节的虚拟阻抗刚度由0.1升至0.12 Nm/rad、阻尼由0.003升至0.0036 Nm·s/rad（+20%）。其余手指、当前输出限额和斜率不变。调参前备份在 `.calibration_backup/thumb-feedback-20261112-033303/`。这些是从手空间的软件虚拟阻抗，不等于手套驱动器MIT的kp/kd。

`wuji-monitor-torque.json` 仅供新监视UI做电流×Kt的估算，不用于控制。手套5.1 Nm/A沿用仿真配置中的空载参考；WuJi各轴默认null，未标定通道保持缺失。可通过 `pixi run teleop-ui --torque-config <文件>` 指定显示换算配置。源系数和来源说明随JSONL历史保存。

新驱动阻抗配置为 `wuji-driver-impedance-right.json`，`force-contact` 默认启用；旧Tianji虚拟阻抗仍用于接触判据/torque回退，不和新PD强度重复相加。源仿真J1倍率1.5经用户拇指增强20%后为1.8，J2–J4为1.2；当前J1/J2名义上限各10 Nm，其余各4 Nm。

## J1/J2名义上限调整

2026-11-13 按用户要求，将J1从4 Nm、J2从2 Nm均改为5 Nm，J3–J12保持4 Nm。`driver_impedance`保留0.85预算系数，因此J1/J2的提交名义预算约4.25 Nm，其余约3.4 Nm，另扣编码余量；不是实体硬力矩上限。硬度、kp/kd、摩擦补偿、零位及释放保护不变。该限额文件由接触阻抗和线性力矩入口共用，下次启动控制时加载。备份：`.calibration_backup/thumb-torque-5nm-20261113-021726`。

2026-11-13 后续按用户要求，将J1/J2从各5 Nm进一步调至各10 Nm，其余各4 Nm；0.85系数保留，提交名义预算约8.5/3.4 Nm，另扣编码余量。此变更尚未进行实机验证；022325采样属于先前5 Nm配置，不能用于验证10 Nm。下次启动加载。备份：`.calibration_backup/thumb-torque-10nm-20261113-024205`。
