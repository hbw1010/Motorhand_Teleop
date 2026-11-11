#include "HandClientG4.hpp"
#include <iostream>
#include <cmath>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_stop_requested{false};
void SignalHandler(int) {
    g_stop_requested = true;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    const std::string port = (argc > 1) ? argv[1] : "/dev/ttyUSB0";
    std::cout << ">> [STM32G4 网关方案] 连接设备: " << port << " @ 200 Hz...\n";
    
    lqs::HandClientG4 hand;
    if (!hand.Connect(port, 921600, 200)) {
        std::cerr << "打开失败，请确认串口号或执行: sudo chmod 666 " << port << '\n';
        return 1;
    }
    std::cout << ">> 连接成功，下发 200Hz 闭环正弦测试 (支持 Ctrl+C 安全停止)...\n";
    auto t0 = std::chrono::steady_clock::now();

    while (!g_stop_requested) {
        auto now = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(now - t0).count();
        if (t >= 5.0) break;

        float deg = 20.0f * std::sin(2.0f * 3.14159f * 0.5f * t);
        hand.SetServoCommand(1, 9, deg, 60.0f, 1.0f);
        hand.SetMitCommand(2, 5, deg, 15.0f, 0.5f);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::cout << "\n>> 测试结束（或触发中断），正在执行 12 轴全量安全卸力...\n";
    hand.Disconnect();
    std::cout << ">> 灵巧手已平滑释放，安全退出。\n";
    return 0;
}
