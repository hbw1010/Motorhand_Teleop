# 手部力反馈资产

- `max-torque-linear-right.json` 和 `contact-impedance-right.json`：当前天机右手参数。文件中的力矩和电流是名义估计/命令上限，不是实测接触力标定。
- `max-torque-linear-left.template.json` 和 `contact-impedance-left.template.json`：左手模板。使用前必须填入左手 Wuji 序列号，并在低负载硬件上重新验证电流、限幅、方向和接触阈值。

C++ worker 会核对限额文件中的 `wuji_serial_number`。不匹配时不得绕过检查；应先确认硬件身份和参数来源。
