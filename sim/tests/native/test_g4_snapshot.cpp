// 用伪终端运行真实 SDK，固定复现 IO 已收到反馈、watchdog 尚未调度的窗口。
#include "g4_shim.cpp"

#include <pty.h>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct PseudoSerial {
    int master = -1, slave = -1;
    char path[128]{};
    PseudoSerial() {
        Check(openpty(&master, &slave, path, nullptr, nullptr) == 0, "无法创建伪串口");
    }
    ~PseudoSerial() { close(master); close(slave); }
};

void TestSnapshotBeforeWatchdog() {
    PseudoSerial port;
    Shim shim;
    // 故意不启动 watchdog，模拟 g4_open 返回后该线程还没有获得 CPU。
    Check(shim.client.Connect(port.path), "无法连接伪串口");
    G4Joint joints[12]{};
    Check(g4_snapshots(&shim, joints, 12) == 12, "必须包含 12 条固定路由");
    for (const auto& joint : joints) {
        Check(joint.samples == 0 && joint.received_ns == 0, "无反馈时不能制造在线时间");
    }

    const auto sent_ns = MonotonicNs();
    for (uint16_t id = 1; id <= 12; ++id) {
        auto frame = gateway::Frame(0x90, id <= 4 ? 3 : (id <= 8 ? 2 : 1), id);
        frame.len = 8;
        frame.data[0] = 0x40;  // 带位置的 type-2 反馈，角度为 0。
        gateway::Seal(frame);
        Check(write(port.master, &frame, sizeof(frame)) == sizeof(frame), "注入反馈失败");
    }

    const auto deadline = MonotonicNs() + 2'000'000'000LL;
    while (true) {
        const auto snapshots = shim.client.GetTimedSnapshots();
        bool complete = snapshots.size() == 12;
        for (const auto& [key, timed] : snapshots) complete &= timed.state.samples == 1;
        if (complete) break;
        Check(MonotonicNs() < deadline, "SDK 未收到全部伪串口反馈");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    Check(g4_snapshots(&shim, joints, 12) == 12, "反馈路由缺失");
    const auto now = MonotonicNs();
    std::map<uint32_t, int64_t> timestamps;
    const auto timed = shim.client.GetTimedSnapshots();
    for (const auto& joint : joints) {
        Check(joint.samples == 1 && (joint.valid & 1), "位置反馈必须有效");
        Check(joint.received_ns != 0, "已收到位置反馈，但时间戳为 0：启动时误判全轴离线");
        Check(joint.received_ns >= sent_ns - 1000 && joint.received_ns <= now,
              "接收时间必须属于本次反馈且与 CLOCK_MONOTONIC 同源");
        Check(joint.received_ns == int64_t(timed.at({uint8_t(joint.bus), uint16_t(joint.id)}).position_received_us) * 1000,
              "角度和时间必须来自同一 SDK 快照");
        timestamps[joint.id] = joint.received_ns;
    }

    // 只消费旧快照不能续期；超过在线阈值后依然应当判为离线。
    std::this_thread::sleep_for(std::chrono::milliseconds(270));
    g4_snapshots(&shim, joints, 12);
    for (const auto& joint : joints) {
        Check(joint.received_ns == timestamps.at(joint.id), "旧快照不能刷新接收时间");
        Check(MonotonicNs() - joint.received_ns > 250'000'000LL, "断流必须保留过期状态");
    }
}
}  // namespace

int main() {
    try { TestSnapshotBeforeWatchdog(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
