#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

#pragma pack(push, 1)
struct GatewayWireFrame {
    uint8_t header[2], version, kind;
    uint16_t sequence;
    uint8_t bus, len;
    uint16_t can_id, flags;
    uint32_t timestamp_us;
    uint8_t data[8];
    uint32_t auxiliary;
    uint16_t detail, crc;
};
#pragma pack(pop)
static_assert(sizeof(GatewayWireFrame) == 32, "Wire ABI");

struct MotorFeedback {
    float angle_deg = 0, current_a = 0, temp_c = 0, speed_rpm = 0;
    uint8_t err_code = 0;
    uint32_t timestamp_us = 0, valid = 0;
    uint64_t position_time = 0, current_time = 0, temperature_time = 0, speed_time = 0;
};

struct MotorCommand {
    bool enabled = false, is_mit_mode = false;
    float target_angle = 0, servo_spd = 30, servo_cur = 1;
    float mit_kp = 15, mit_kd = .5f, mit_vel = 0, mit_tor = 0;
};

// Plain C ABI, mirrored by ctypes. Hand_SnapshotSize validates alignment at load.
struct HandSnapshot {
    uint32_t bus, id, valid, online, error;
    float angle_deg, current_a, temp_c, speed_rpm;
    float age_ms, current_age_ms, speed_age_ms, temperature_age_ms, hz;
    uint64_t samples;
    uint32_t board_timestamp_us, max_board_gap_us, max_host_gap_us;
    uint32_t host_over_10ms, board_over_10ms, tx_failures;
};

// 项目扩展：不更改HandSnapshot的C ABI布局，单独携带SDK解析位置包的主机steady时间。
struct TimedHandSnapshot {
    HandSnapshot state{};
    uint64_t position_received_us{}, current_received_us{};
};

struct HandDiagnostics {
    uint64_t crc_errors = 0, invalid_frames = 0, received_can = 0;
    uint64_t tx_failures = 0, serial_errors = 0, skipped_cycles = 0;
    uint64_t gateway_crc = 0, gateway_drops = 0, gateway_uart_errors = 0;
    uint32_t connected = 0, mode = 0, target_hz = 200, routes = 0;
    uint32_t telemetry_enabled = 0;
};

namespace gateway {
inline uint64_t NowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline uint32_t Key(unsigned bus, unsigned id) { return (bus << 16) | id; }
inline uint16_t Crc(const uint8_t* data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = static_cast<uint16_t>((crc << 1) ^ ((crc & 0x8000) ? 0x1021 : 0));
    }
    return crc;
}
inline GatewayWireFrame Frame(uint8_t kind, uint8_t bus = 0, uint16_t id = 0) {
    GatewayWireFrame frame{};
    frame.header[0] = 0xA5; frame.header[1] = 0x5A;
    frame.version = 1; frame.kind = kind; frame.bus = bus; frame.can_id = id;
    return frame;
}
inline void Seal(GatewayWireFrame& frame) {
    frame.crc = Crc(reinterpret_cast<const uint8_t*>(&frame), 30);
}
inline GatewayWireFrame Query(uint8_t bus, uint16_t id, uint8_t parameter = 1) {
    auto frame = Frame(2, bus, id);
    frame.len = 2; frame.data[0] = 0xE0; frame.data[1] = parameter; frame.auxiliary = 10;
    return frame;
}

class Parser {
public:
    uint64_t crc_errors = 0, invalid_frames = 0;
    size_t Buffered() const { return end_ - begin_; }
    template<class Consumer>
    void Feed(const uint8_t* input, size_t size, Consumer consume) {
        while (size) {
            if (end_ == buffer_.size()) Compact();
            size_t count = std::min(size, buffer_.size() - end_);
            std::memcpy(buffer_.data() + end_, input, count);
            end_ += count; input += count; size -= count;
            while (end_ - begin_ >= sizeof(GatewayWireFrame)) {
                const auto* bytes = buffer_.data() + begin_;
                if (bytes[0] != 0xA5 || bytes[1] != 0x5A || bytes[2] != 1) {
                    ++begin_;
                    continue;
                }
                uint16_t crc;
                std::memcpy(&crc, bytes + 30, 2);
                if (Crc(bytes, 30) != crc) { ++crc_errors; ++begin_; continue; }
                GatewayWireFrame frame;
                std::memcpy(&frame, bytes, sizeof(frame));
                begin_ += sizeof(frame);
                if (frame.len > 8 || (frame.kind == 0x90 &&
                    (frame.bus < 1 || frame.bus > 3 || frame.can_id > 0x7FF))) {
                    ++invalid_frames;
                    continue;
                }
                consume(frame);
            }
            if (begin_ == end_ || begin_ > buffer_.size() / 2) Compact();
        }
    }
private:
    std::array<uint8_t, 4096> buffer_{};
    size_t begin_ = 0, end_ = 0;
    void Compact() {
        const size_t remaining = end_ - begin_;
        std::memmove(buffer_.data(), buffer_.data() + begin_, remaining);
        begin_ = 0; end_ = remaining;
    }
};

enum Field : uint32_t { Position = 1, Current = 2, Temperature = 4, Speed = 8 };
inline float BigFloat(const uint8_t* bytes) {
    uint32_t bits = uint32_t(bytes[0]) << 24 | uint32_t(bytes[1]) << 16 |
                    uint32_t(bytes[2]) << 8 | bytes[3];
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
inline uint32_t Decode(const GatewayWireFrame& frame, MotorFeedback& feedback, uint64_t now) {
    auto next = feedback;
    uint32_t fields = 0;
    uint8_t type = frame.data[0] >> 5;
    if (type == 5 && frame.len == 6) {
        const float value = BigFloat(frame.data + 2);
        if (!std::isfinite(value)) return 0;
        switch (frame.data[1]) {
        case 1: next.angle_deg = value; fields = Position; break;
        case 2: next.speed_rpm = value; fields = Speed; break;
        case 3: next.current_a = value; fields = Current; break;
        default: return 0;
        }
    } else if (type == 2 && frame.len == 8) {
        next.angle_deg = BigFloat(frame.data + 1);
        if (!std::isfinite(next.angle_deg)) return 0;
        const auto current = static_cast<int16_t>(uint16_t(frame.data[5]) << 8 | frame.data[6]);
        next.current_a = current / 100.f;
        fields = Position | Current;
        if (frame.data[7] >= 50 && frame.data[7] <= 220) {
            next.temp_c = (int(frame.data[7]) - 50) / 2.f;
            fields |= Temperature;
        } else next.valid &= ~Temperature;
    } else if (type == 1 && frame.len == 8) {
        const uint16_t position = uint16_t(frame.data[1]) << 8 | frame.data[2];
        const uint16_t current = uint16_t(frame.data[4] & 15) << 8 | frame.data[5];
        next.angle_deg = (-5.f + position / 65535.f * 10.f) * 180.f / 3.14159265f;
        next.current_a = -2.f + current / 4095.f * 4.f;
        fields = Position | Current;
        if (frame.data[6] >= 50 && frame.data[6] <= 220) {
            next.temp_c = (int(frame.data[6]) - 50) / 2.f;
            fields |= Temperature;
        } else next.valid &= ~Temperature;
    } else return 0;
    next.err_code = frame.data[0] & 31;
    if (fields & Position) { next.position_time = now; next.timestamp_us = frame.timestamp_us; }
    if (fields & Current) next.current_time = now;
    if (fields & Temperature) next.temperature_time = now;
    if (fields & Speed) next.speed_time = now;
    next.valid |= fields;
    feedback = next;
    return fields;
}
inline float Age(uint64_t now, uint64_t timestamp) {
    return timestamp ? float(now - timestamp) / 1000.f : -1.f;
}

struct ReleaseTracker {
    bool pending = false, failed = false;
    unsigned attempts = 0;
    uint64_t settle_until = 0;
    std::array<uint16_t, 3> sequences{};
    void Begin() { *this = {}; pending = true; }
    bool NeedsSend() const { return pending && attempts < sequences.size(); }
    void Submitted(uint16_t sequence, uint64_t now) {
        if (!NeedsSend()) return;
        sequences[attempts++] = sequence;
        settle_until = now + 20000;
    }
    void Failure(uint16_t sequence) {
        if (!pending) return;
        for (unsigned index = 0; index < attempts; ++index)
            if (sequences[index] == sequence) failed = true;
    }
    bool Settled(uint64_t now) const { return pending && attempts == sequences.size() && now >= settle_until; }
};

struct RouteState {
    uint8_t bus;
    uint16_t id;
    MotorFeedback feedback;
    MotorCommand command;
    ReleaseTracker release;
    uint64_t telemetry_due_us = 0;
    uint8_t telemetry_parameter = 2;
    uint64_t samples = 0, window_start = 0, window_samples = 0;
    uint32_t max_board_gap = 0, max_host_gap = 0, host_over10 = 0, board_over10 = 0;
    uint32_t tx_failures = 0;
    float hz = 0;
    RouteState(uint8_t bus_value, uint16_t id_value) : bus(bus_value), id(id_value) {}
    bool Accept(const GatewayWireFrame& frame, uint64_t now) {
        if (frame.kind != 0x90 || frame.bus != bus || frame.can_id != id) return false;
        const uint64_t previous_host = feedback.position_time;
        const uint32_t previous_board = feedback.timestamp_us;
        const uint32_t fields = Decode(frame, feedback, now);
        if (!fields) return false;
        if (!(fields & Position)) return true;
        if (samples) {
            const uint32_t board_gap = frame.timestamp_us - previous_board;
            const uint32_t host_gap = static_cast<uint32_t>(
                std::min<uint64_t>(now - previous_host, UINT32_MAX));
            max_board_gap = std::max(max_board_gap, board_gap);
            max_host_gap = std::max(max_host_gap, host_gap);
            host_over10 += host_gap > 10000;
            board_over10 += board_gap > 10000;
        }
        ++samples;
        if (!window_start) { window_start = now; window_samples = samples; }
        if (now - window_start >= 500000) {
            hz = float(samples - window_samples) * 1000000.f / float(now - window_start);
            window_start = now; window_samples = samples;
        }
        return true;
    }
    TimedHandSnapshot TimedSnapshot(uint64_t now) const {
        return {Snapshot(now), feedback.position_time, feedback.current_time};
    }
    HandSnapshot Snapshot(uint64_t now) const {
        HandSnapshot result{};
        result.bus = bus; result.id = id; result.valid = feedback.valid; result.error = feedback.err_code;
        result.angle_deg = feedback.angle_deg; result.current_a = feedback.current_a;
        result.temp_c = feedback.temp_c; result.speed_rpm = feedback.speed_rpm;
        result.age_ms = Age(now, feedback.position_time);
        result.current_age_ms = Age(now, feedback.current_time);
        result.speed_age_ms = Age(now, feedback.speed_time);
        result.temperature_age_ms = Age(now, feedback.temperature_time);
        result.online = (feedback.valid & Position) && result.age_ms <= 50;
        result.hz = result.online ? hz : 0;
        result.samples = samples; result.board_timestamp_us = feedback.timestamp_us;
        result.max_board_gap_us = max_board_gap; result.max_host_gap_us = max_host_gap;
        result.host_over_10ms = host_over10; result.board_over_10ms = board_over10;
        result.tx_failures = tx_failures;
        return result;
    }
};

inline bool CanSupplement(const RouteState& route, uint32_t latest_board_us, uint64_t now) {
    return (route.feedback.valid & Position) && now >= route.telemetry_due_us &&
        now - route.feedback.position_time <= 1000 &&
        uint32_t(latest_board_us - route.feedback.timestamp_us) <= 1000;
}

inline void StoreWord(uint64_t word, uint8_t* data) {
    for (unsigned i = 0; i < 8; ++i) data[i] = static_cast<uint8_t>(word >> (56 - i * 8));
}
inline void PackServo(float position, float speed, float current, uint8_t* data) {
    uint32_t position_bits;
    std::memcpy(&position_bits, &position, 4);
    uint16_t velocity = uint16_t(std::round(std::clamp(speed, 0.f, 3276.7f) * 10)) & 0x7FFF;
    uint16_t limit = uint16_t(std::round(std::clamp(current, 0.f, 409.5f) * 10)) & 0xFFF;
    StoreWord(uint64_t(1) << 61 | uint64_t(position_bits) << 29 |
              uint64_t(velocity) << 14 | uint64_t(limit) << 2 | 2, data);
}
inline void PackMit(float position, float velocity, float kp, float kd, float torque, uint8_t* data) {
    uint16_t p = uint16_t(std::round((std::clamp(position, -5.f, 5.f) + 5) / 10 * 65535));
    uint16_t v = uint16_t(std::round((std::clamp(velocity, -10.f, 10.f) + 10) / 20 * 4095));
    uint16_t k = uint16_t(std::round(std::clamp(kp, 0.f, 500.f) / 500 * 4095));
    uint16_t d = uint16_t(std::round(std::clamp(kd, 0.f, 5.f) / 5 * 511));
    uint16_t t = uint16_t(std::round((std::clamp(torque, -10.f, 10.f) + 10) / 20 * 4095));
    StoreWord(uint64_t(k) << 49 | uint64_t(d) << 40 | uint64_t(p) << 24 | uint64_t(v) << 12 | t, data);
}
inline bool ValidCommand(const MotorCommand& command) {
    return std::isfinite(command.target_angle) && std::isfinite(command.servo_spd) &&
        std::isfinite(command.servo_cur) && std::isfinite(command.mit_kp) &&
        std::isfinite(command.mit_kd) && std::isfinite(command.mit_vel) &&
        std::isfinite(command.mit_tor);
}
inline GatewayWireFrame CommandFrame(uint8_t bus, uint16_t id, const MotorCommand& command) {
    if (!command.enabled) return Query(bus, id);
    auto frame = Frame(2, bus, id);
    frame.len = 8; frame.auxiliary = 10;
    if (command.is_mit_mode)
        PackMit(command.target_angle * 3.14159265f / 180.f, command.mit_vel,
                command.mit_kp, command.mit_kd, command.mit_tor, frame.data);
    else PackServo(command.target_angle, command.servo_spd, command.servo_cur, frame.data);
    return frame;
}
inline GatewayWireFrame Release(uint8_t bus, uint16_t id) {
    auto frame = Frame(2, bus, id);
    frame.len = 8; frame.auxiliary = 10;
    PackMit(0, 0, 0, 0, 0, frame.data);
    return frame;
}
}
