# STM32G4 3-CAN 集中式网关 (LQS-GW02) Linux C++ SDK

用于博文进行算法控制及与原 USB-to-CAN 方案进行横向性能对比测试。

## 一、 硬件与接口
- **老方案（USB-to-CAN）**：插 3 个独立 USB 转接头，走各自总线，多端口分别调度。
- **本方案（STM32G4 集中式网关）**：仅需单个 Type-C 串口 (/dev/ttyUSB0 或 /dev/ttyACM0)，921600 波特率，板载 G4 硬件并发调度 3 路 CAN，实测 200Hz 闭环稳定。

## 二、 Linux 下编译与运行 (Ubuntu / ROS)
1. 授予串口权限：
   sudo chmod 666 /dev/ttyUSB0

2. 编译并运行算法 Demo：
   cd examples
   mkdir build && cd build
   cmake ..
   make
   ./g4_control_demo /dev/ttyUSB0

## 三、 接口与控制规范
- **目标角度**：无论伺服模式还是 MIT 混控模式，传入目标位置统一使用**角度制（°，deg）**。在 MIT 模式下底层驱动会自动换算弧度，上层算法严禁重复转换。
- **安全卸力**：程序退出或异常中断时，SDK 会自动下发 3 轮零刚度平滑卸力报文，防止断电砸手。


