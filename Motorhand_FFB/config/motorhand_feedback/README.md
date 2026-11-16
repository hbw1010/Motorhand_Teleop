# 手部力反馈资产

## 本文档范围

所属分支：`sim`。原 sim2.0 的双工程整合版本；外层 sim 与 Motorhand_FFB 使用各自环境，运行目录不能混用。

本目录为 Motorhand_FFB 手部适配器的参数与模板；与外层 sim/config/default.yaml 及根目录天机参数分别管理。

[返回仓库首页与分支对照](../../../README.md)

- `max-torque-linear-right.json` 和 `contact-impedance-right.json`：当前天机右手参数。文件中的力矩和电流是名义估计/命令上限，不是实测接触力标定。
- `max-torque-linear-left.template.json` 和 `contact-impedance-left.template.json`：左手模板。使用前必须填入左手 Wuji 序列号，并在低负载硬件上重新验证电流、限幅、方向和接触阈值。

C++ worker 会核对限额文件中的 `wuji_serial_number`。不匹配时不得绕过检查；应先确认硬件身份和参数来源。
