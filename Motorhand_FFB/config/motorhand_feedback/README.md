# 历史反馈参数与左手模板

所属分支：`sim2.0`，原 `sim3.0`。[返回仓库首页与分支对照](../../../README.md)。

本目录保留手部适配器的历史右手参数及左手模板。当前 `wuji-force-contact` / `wuji-force-linear` 使用工作区根目录 [config/feedback/](../../../config/feedback/README.md) 的 `wuji-*.json`，不读取本目录作为默认运行配置。

- `max-torque-linear-right.json`、`contact-impedance-right.json`：保留的右手参数来源，不能与当前已调整的 Wuji 配置混为一套。
- `max-torque-linear-left.template.json`、`contact-impedance-left.template.json`：左手模板，需要填写对应设备身份并完成方向、零位和限额核对；模板不表示已完成硬件标定。
- `sim` 分支仍有 `Motorhand_FFB/tianji_teleop/` 独立手部 ROS2 运行器；本分支已经移除运行器，目录里保留参数不表示仍可启动该节点。

设备序列号检查由所选运行器及其实际加载的限额文件决定，不应通过修改历史模板绕过身份核对。文件中的电流、软件阻抗和名义 Nm 不是实测接触力标定。
