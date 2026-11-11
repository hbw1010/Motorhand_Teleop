# LQS GW02 三路 CAN 上位机

## 运行

需要 Windows x64、Python x64、PyQt5 5.15.11、pyserial 3.5，以及已烧录 LQS-GW02 的测试板。压缩包包含编译好的 hand_sdk.dll。

解压后，在本目录打开 PowerShell 执行。

```powershell
python -m pip install -r requirements.txt
python gui_test_joints_cpp.py
```

选择实际串口后点击连接。本次使用 COM84、921600 波特率，COM74 为 J-Link。程序自动扫描 CAN1–3 的 ID1–16，最多挂载 12 条总线与 ID 路由。每路电机数量和 ID 顺序可以不同。

默认每台 200 Hz 只读位置采集，并显示电机错误码。连接扫描发送 E0 01。需要控制时，在单台卡片选择伺服位置或 MIT 力位混控，再显式使能。修改目标和参数会更新原生通信线程中的最新命令。

勾选“补充电流/速度”后，按对应电机的新位置反馈时机逐条发送低频查询，每项约 5 Hz。该选项默认关闭，附加查询实测仍有少量 CAN 发送失败。12 轴使用 230/235 Hz 时自动暂停补充查询。温度只从带温度的控制反馈更新。未取得、已过期字段显示 `--`。

## 状态与控制行为

- 每台显示位置反馈频率、数据年龄和错误状态。位置超过 50 ms 未更新时标记离线，保留最后角度并变灰。界面约 30 Hz 刷新。
- 鼠标悬停在频率文字上可查看主机与板端最大反馈间隔。频率只统计有效位置反馈。
- 主机接收 CRC 为当前 SDK 会话计数。网关 CRC 与丢帧为固件累计计数，判断本轮异常需要比较前后增量。本次故意注入 100 个 CRC 错包后，网关累计 CRC 从 300 增至 400。
- 使能后切换为主机 5 ms 周期控制。目标按最新值发送，错过周期会跳过并计数。
- 使能被原生 SDK 拒绝时，界面撤回 ON 状态，并清除已排队的旧使能请求。
- 全部软释放或断开连接会对本次已使能电机发送三次零增益、零前馈力矩 MIT 报文，并检查网关拒绝和异步发送失败。失败会通过 LastError 和界面错误提示反馈。机械卸力效果需要现场确认。
- 电机反馈离线后，已使能电机仍沿用原工程的持续发送既有目标行为。50 ms 用于反馈过期判定。显式软释放和断开连接可以终止该控制会话。
- 本次系统验收使用只读查询。人工操作过的首次测试单独标注，受控运动精度与机械卸力尚未专项验收。控制量程沿用原工程配置。
- 标定按 `CAN:ID` 保存。同 ID 跨总线可分别标定。旧 ID 标定仅在扫描结果中该 ID 唯一时迁移使用。

## SDK 接口

Python 入口为 hand_api.py，调用顺序如下。

```python
from hand_api import HandApi

hand = HandApi()
try:
    hand.connect('COM84')
    hand.set_rate(200)
    states = hand.snapshots()
    diagnostics = hand.diagnostics()
    print(states)
finally:
    hand.close()
```

主要 C 导出包括 Hand_Init、Hand_Close、Hand_SetCmdBus、Hand_GetSnapshot、Hand_GetDiagnostics、Hand_GetLastError、Hand_SetRate、Hand_SetTelemetry、Hand_ResetMetrics。结构体定义见 GatewayProtocol.hpp，Python 绑定会核对结构体大小。每个进程中的 HandApi 实例共享一个 DLL 会话。

旧 Hand_SetCmd、Hand_GetFb、Hand_GetDiscoveredRoutes、Hand_GetActualHz 保留。ID 跨总线重复时，旧 ID-only 命令会拒绝更新，GetFb 返回 false。Hand_GetActualHz 现在返回各路位置频率的最低值。

## 编译与测试

需要 Zig C++ 或 MinGW g++ 的 Windows x64 C++17 编译器。运行已提供的 DLL 无需编译器。

```powershell
python build_sdk.py --compiler 'C:\tools\zig\zig.exe'
python -m unittest discover -s tests -p 'test_*.py'
python tests/measure_native.py --port COM84 --hz 200 --seconds 120 --gui
python tests/measure_native.py --port COM84 --hz 235 --seconds 30
```

编译前关闭使用 DLL 的所有程序。build_sdk.py 会编译 DLL、只读监视器和三组 C++ 回归程序，然后运行回归并安装 DLL。打包版本的 tests/support/gateway_sdk.py 来自现有固件工程，用于独立的链路核验。

```powershell
python tests/verify_gateway_link.py --port COM84 --routes test_results/native_20261110_211855/summary.json
```

measure_native.py 的 GUI 模式会禁用交互控制，保证只读验收。可增加 `--telemetry` 比较附加状态查询。每次测试保存原始快照、汇总 JSON，GUI 测试同时保存截图。

详细数据和实施记录见 docs/优化与测试报告_20261110.md。
