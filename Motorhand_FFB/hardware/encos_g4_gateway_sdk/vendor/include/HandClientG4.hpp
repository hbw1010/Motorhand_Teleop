#pragma once
#include "GatewayProtocol.hpp"
#include "PosixSerial.hpp"
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <chrono>

namespace lqs {
struct JointKey {
    uint8_t bus;
    uint16_t id;
    bool operator<(const JointKey& o) const {
        return (bus != o.bus) ? (bus < o.bus) : (id < o.id);
    }
};

class HandClientG4 {
public:
    HandClientG4() = default;
    ~HandClientG4() { Disconnect(); }

    bool Connect(const std::string& port = "/dev/ttyUSB0", int baud = 921600, unsigned hz = 200) {
        Disconnect();
        if (!serial_.Open(port, baud)) return false;
        target_hz_ = hz;
        running_ = true;
        io_thread_ = std::thread(&HandClientG4::IoLoop, this);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return true;
    }

    void Disconnect() {
        if (running_) {
            running_ = false;
            if (io_thread_.joinable()) io_thread_.join();
            SendReleaseBatchAll();
        }
        serial_.Close();
    }

    void SetServoCommand(uint8_t bus, uint16_t id, float deg, float spd = 50.0f, float cur = 1.0f) {
        MotorCommand cmd;
        cmd.enabled = true; cmd.is_mit_mode = false;
        cmd.target_angle = deg; cmd.servo_spd = spd; cmd.servo_cur = cur;
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        commands_[{bus, id}] = cmd;
    }

    void SetMitCommand(uint8_t bus, uint16_t id, float deg, float kp, float kd, float vel = 0.0f, float tor = 0.0f) {
        MotorCommand cmd;
        cmd.enabled = true; cmd.is_mit_mode = true;
        cmd.target_angle = deg; cmd.mit_kp = kp; cmd.mit_kd = kd; cmd.mit_vel = vel; cmd.mit_tor = tor;
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        commands_[{bus, id}] = cmd;
    }

    void DisableJoint(uint8_t bus, uint16_t id) {
        MotorCommand cmd;
        cmd.enabled = false;
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        commands_[{bus, id}] = cmd;
    }

    std::map<JointKey, HandSnapshot> GetSnapshots() {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        return latest_snapshots_;
    }

private:
    PosixSerial serial_;
    std::atomic<bool> running_{false};
    unsigned target_hz_ = 200;
    std::thread io_thread_;
    std::mutex cmd_mutex_;
    std::map<JointKey, MotorCommand> commands_;
    std::mutex snapshot_mutex_;
    std::map<JointKey, HandSnapshot> latest_snapshots_;

    void SendReleaseBatchAll() {
        static const JointKey all_joints[] = {
            {1, 9}, {1, 10}, {1, 11}, {1, 12},
            {2, 5}, {2, 6},  {2, 7},  {2, 8},
            {3, 1}, {3, 2},  {3, 3},  {3, 4}
        };

        std::array<GatewayWireFrame, 12> rel_batch;
        for (size_t i = 0; i < 12; ++i) {
            rel_batch[i] = gateway::Release(all_joints[i].bus, all_joints[i].id);
            gateway::Seal(rel_batch[i]);
        }

        for (int r = 0; r < 3; ++r) {
            serial_.Write(rel_batch.data(), sizeof(rel_batch));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    void IoLoop() {
        gateway::Parser parser;
        uint8_t rx[1024];
        std::map<JointKey, gateway::RouteState> routes;
        auto add_route = [&](uint8_t b, uint16_t id) { routes.emplace(JointKey{b, id}, gateway::RouteState(b, id)); };
        for (uint16_t id = 9; id <= 12; ++id) add_route(1, id);
        for (uint16_t id = 5; id <= 8; ++id) add_route(2, id);
        for (uint16_t id = 1; id <= 4; ++id) add_route(3, id);

        const auto cycle = std::chrono::microseconds(1000000 / target_hz_);
        uint16_t sequence = 0;

        while (running_) {
            auto t0 = std::chrono::steady_clock::now();
            uint64_t now_us = gateway::NowUs();

            std::array<GatewayWireFrame, 12> tx_batch;
            size_t tx_count = 0;
            {
                std::lock_guard<std::mutex> lock(cmd_mutex_);
                for (auto& [k, route] : routes) {
                    MotorCommand c;
                    if (commands_.count(k)) c = commands_[k];
                    auto f = gateway::CommandFrame(k.bus, k.id, c);
                    f.sequence = ++sequence;
                    gateway::Seal(f);
                    tx_batch[tx_count++] = f;
                }
            }
            serial_.Write(tx_batch.data(), tx_count * sizeof(GatewayWireFrame));

            int n = 0;
            while ((n = serial_.Read(rx, sizeof(rx))) > 0) {
                parser.Feed(rx, n, [&](const GatewayWireFrame& f) {
                    JointKey k{f.bus, f.can_id};
                    if (routes.count(k)) routes.at(k).Accept(f, gateway::NowUs());
                });
            }

            {
                std::lock_guard<std::mutex> lock(snapshot_mutex_);
                for (const auto& [k, route] : routes) {
                    latest_snapshots_[k] = route.Snapshot(now_us);
                }
            }

            std::this_thread::sleep_until(t0 + cycle);
        }
    }
};
}