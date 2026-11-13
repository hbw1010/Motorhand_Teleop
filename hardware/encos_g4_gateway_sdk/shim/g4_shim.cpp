// 厂商SDK的C ABI包装（vendor/include/HandClientG4.hpp）；时间快照扩展见vendor/LOCAL_CHANGES.md。
// for "驱动 2" (STM32G4 LQS-GW02 gateway).  Loaded from Python with ctypes.
//
// Only adds things *above* the SDK:
//  * command deadline watchdog: when the controller stops refreshing a
//    command, the joint gets the SDK's zero-gain MIT command (same as the
//    vendor release frame) and then DisableJoint, instead of the SDK
//    re-sending a stale MIT target forever;
//  * a host receive timestamp per joint, taken when the SDK's sample counter
//    advances（保留此旧C ABI路径的消费时间语义；原生阻抗使用SDK的GetTimedSnapshots真实解析时间）。
#include "HandClientG4.hpp"

#include <time.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>

namespace {
int64_t MonotonicNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);  // == Python time.monotonic_ns()
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct Pending {
    int64_t deadline_ns = 0;       // 0: no deadline
    int64_t disable_at_ns = 0;     // >0: limp MIT sent, DisableJoint at this time
};

struct Seen {
    uint64_t samples = 0;
    int64_t received_ns = 0;
};

struct Shim {
    lqs::HandClientG4 client;
    std::mutex mutex;
    std::map<lqs::JointKey, Pending> pending;
    std::map<lqs::JointKey, Seen> seen;
    std::atomic<bool> running{false};
    std::thread watchdog;
    uint64_t releases = 0;

    void Limp(const lqs::JointKey& k, Pending& p, int64_t now) {
        client.SetMitCommand(k.bus, k.id, 0.f, 0.f, 0.f, 0.f, 0.f);
        p.deadline_ns = 0;
        p.disable_at_ns = now + 20000000;  // >= 3 SDK cycles of zero-gain MIT
        ++releases;
    }

    void Loop() {
        while (running) {
            const int64_t now = MonotonicNs();
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (auto& [k, p] : pending) {
                    if (p.deadline_ns && now >= p.deadline_ns) Limp(k, p, now);
                    if (p.disable_at_ns && now >= p.disable_at_ns) {
                        client.DisableJoint(k.bus, k.id);
                        p.disable_at_ns = 0;
                    }
                }
                for (const auto& [k, s] : client.GetSnapshots()) {
                    auto& entry = seen[k];
                    if (s.samples != entry.samples) {
                        entry.samples = s.samples;
                        entry.received_ns = now;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};
}  // namespace

extern "C" {

struct G4Joint {
    uint32_t bus, id, valid, error;
    float angle_deg, current_a, temp_c, speed_rpm, hz;
    uint64_t samples;
    int64_t received_ns;
    uint32_t board_timestamp_us, max_board_gap_us, max_host_gap_us, tx_failures;
};

uint32_t g4_joint_size() { return sizeof(G4Joint); }
int64_t g4_now_ns() { return MonotonicNs(); }

void* g4_open(const char* port, int baud, unsigned hz) {
    auto* shim = new Shim();
    if (!shim->client.Connect(port, baud, hz)) {
        delete shim;
        return nullptr;
    }
    shim->running = true;
    shim->watchdog = std::thread(&Shim::Loop, shim);
    return shim;
}

void g4_set_mit(void* h, uint32_t bus, uint32_t id, float deg, float kp, float kd, float vel,
                float tor, int64_t deadline_ns) {
    auto* shim = static_cast<Shim*>(h);
    std::lock_guard<std::mutex> lock(shim->mutex);
    shim->client.SetMitCommand(uint8_t(bus), uint16_t(id), deg, kp, kd, vel, tor);
    shim->pending[{uint8_t(bus), uint16_t(id)}] = Pending{deadline_ns, 0};
}

void g4_set_servo(void* h, uint32_t bus, uint32_t id, float deg, float spd, float cur,
                  int64_t deadline_ns) {
    auto* shim = static_cast<Shim*>(h);
    std::lock_guard<std::mutex> lock(shim->mutex);
    shim->client.SetServoCommand(uint8_t(bus), uint16_t(id), deg, spd, cur);
    shim->pending[{uint8_t(bus), uint16_t(id)}] = Pending{deadline_ns, 0};
}

// Zero-gain MIT for a few cycles, then back to read-only query.
void g4_release(void* h, uint32_t bus, uint32_t id) {
    auto* shim = static_cast<Shim*>(h);
    std::lock_guard<std::mutex> lock(shim->mutex);
    lqs::JointKey key{uint8_t(bus), uint16_t(id)};
    shim->Limp(key, shim->pending[key], MonotonicNs());
}

int g4_snapshots(void* h, G4Joint* out, int capacity) {
    auto* shim = static_cast<Shim*>(h);
    const auto snaps = shim->client.GetSnapshots();
    std::lock_guard<std::mutex> lock(shim->mutex);
    int n = 0;
    for (const auto& [k, s] : snaps) {
        if (n >= capacity) break;
        G4Joint& j = out[n++];
        j.bus = s.bus; j.id = s.id; j.valid = s.valid; j.error = s.error;
        j.angle_deg = s.angle_deg; j.current_a = s.current_a; j.temp_c = s.temp_c;
        j.speed_rpm = s.speed_rpm; j.hz = s.hz; j.samples = s.samples;
        auto it = shim->seen.find(k);
        j.received_ns = (it != shim->seen.end() && s.samples) ? it->second.received_ns : 0;
        j.board_timestamp_us = s.board_timestamp_us; j.max_board_gap_us = s.max_board_gap_us;
        j.max_host_gap_us = s.max_host_gap_us; j.tx_failures = s.tx_failures;
    }
    return n;
}

uint64_t g4_release_count(void* h) {
    auto* shim = static_cast<Shim*>(h);
    std::lock_guard<std::mutex> lock(shim->mutex);
    return shim->releases;
}

// SDK Disconnect(): stops the IO thread and sends the vendor 3x safe release.
void g4_close(void* h) {
    auto* shim = static_cast<Shim*>(h);
    shim->running = false;
    if (shim->watchdog.joinable()) shim->watchdog.join();
    shim->client.Disconnect();
    delete shim;
}
}
