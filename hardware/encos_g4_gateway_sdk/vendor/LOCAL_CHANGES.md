# 本项目对SDK的最小扩展

2026-11-12：原生驱动阻抗的速度估计需要位置反馈的真实主机接收/解析时间，不能使用快照消费线程的轮询时间。

- `GatewayProtocol.hpp` 增加独立的 `TimedHandSnapshot` 和 `RouteState::TimedSnapshot()`，携带既有 `MotorFeedback::position_time` 与独立 `current_time`。该值是SDK解析位置反馈时的主机 `steady_clock` 微秒时间，不是电机内部采样时刻。
- `HandClientG4.hpp` 增加 `GetTimedSnapshots()`；位置值、样本计数和时间在同一快照锁中发布。消费旧快照或只收到补充电流帧不会刷新位置时间；仅位置补包也不会刷新电流时间。
- 计算快照年龄的 `NowUs()` 移到本轮串口读取与解析之后，避免本轮新位置时间晚于循环开始时间导致无符号减法下溢。
- 原 `HandSnapshot` 布局（88字节）、`GetSnapshots()`、网关协议及Python shim/C ABI保持兼容。原生控制使用新的快照接口；现有Python调用无需新增参数。

离线测试：`Motorhand_FFB/tests/test_encos_gateway_snapshot_native.cpp`。没有为验证此扩展打开真实串口或启动电机。
