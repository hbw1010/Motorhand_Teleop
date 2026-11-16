# PlotJuggler 只读曲线旁路

所属分支：`sim2.0`，原 `sim3.0`。[返回仓库首页与分支对照](../../README.md)，真机控制入口见 [FFB README](../README.md)。

这些脚本将控制端 UDP 遥测转成 ROS Humble 话题供 PlotJuggler 显示，不接管设备。它们与天机手部控制节点不同：本分支已移除天机 ROS2 运行器，仍可保留这条只读显示旁路。

| 工具 | 用途与依赖 |
|---|---|
| `pixi run teleop-ui` | 新版只读监视器，含反馈链路诊断；不依赖 ROS |
| `pixi run force-ui` | 旧版 Qt 反馈曲线；不依赖 ROS |
| `pixi run plotjuggler-bridge` | 系统 ROS Humble / Python 3.10 下的 UDP → ROS 旁路 |
| `pixi run plotjuggler-pinky` | 使用小指布局打开 PlotJuggler |

在工作区根目录分别启动：

```bash
# 已配置身份/固件的控制命令额外指定 --curves-port 8766
pixi run plotjuggler-bridge --forward-port 8765
pixi run plotjuggler-pinky
# 可选：同时查看转发到 8765 的原生遥测
pixi run teleop-ui
```

桥接默认接收 UDP 8766，`--forward-port 8765` 向原生查看器转发。使用 `pixi run plotjuggler-demo` 可产生演示数据，不连接设备；演示数据不能用于硬件验收。不要同时让多个接收程序抢占同一端口。

ROS 工具使用系统环境，脚本会隔离 Pixi 的 Python/Qt 路径；运行需要 `/opt/ros/humble` 和 PlotJuggler。关闭任一显示窗口不会停止真机控制。
