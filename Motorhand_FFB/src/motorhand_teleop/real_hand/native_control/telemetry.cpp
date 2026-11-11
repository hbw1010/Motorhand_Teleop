#include "telemetry.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace motorhand::native {
namespace {
using Json = nlohmann::json;
constexpr std::array<const char*, 12> glove_names = {
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_pip", "middle_mcp_flex", "middle_pip",
    "ring_mcp_flex", "ring_pip", "pinky_mcp_flex", "pinky_pip"};
constexpr std::array<const char*, 20> wuji_names = {
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_mcp_abd", "index_pip", "index_dip",
    "middle_mcp_flex", "middle_mcp_abd", "middle_pip", "middle_dip",
    "ring_mcp_flex", "ring_mcp_abd", "ring_pip", "ring_dip",
    "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip"};
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
Json number(double x) { return std::isfinite(x) ? Json(x) : Json(nullptr); }
Json stamp(std::uint64_t x) { return x ? Json(x) : Json(nullptr); }
template <std::size_t N> Json values(const std::array<double, N>& input, bool valid = true) {
    Json out = Json::array();
    for (double x : input) out.push_back(valid ? number(x) : Json(nullptr));
    return out;
}
const char* phase(ContactPhase p) {
    switch (p) {
    case ContactPhase::free: return "free";
    case ContactPhase::arming: return "arming";
    case ContactPhase::contact: return "contact";
    case ContactPhase::releasing: return "releasing";
    case ContactPhase::waiting_geometry: return "waiting_geometry";
    case ContactPhase::not_configured: return "not_configured";
    }
    return "not_configured";
}
std::string session_id() {
    std::random_device random;
    static constexpr char digits[] = "0123456789abcdef";
    std::string id; id.reserve(32);
    for (int i = 0; i < 16; ++i) {
        const unsigned byte = random() & 255;
        id.push_back(digits[byte >> 4]); id.push_back(digits[byte & 15]);
    }
    return id;
}
} // namespace

struct TelemetryPublisher::Impl {
    struct Sample { SessionTick tick{}; std::uint64_t sequence{}; };
    std::array<Sample, 3> slots{};
    // SPSC triple buffer: producer owns back, consumer owns front, and
    // atomic middle exchanges ownership. Bit 2 marks an unpublished sample.
    std::atomic<unsigned> middle{1};
    unsigned back = 2, front = 0;
    std::uint64_t sequence = 0;
    std::atomic<std::uint64_t> dropped{0};
    std::atomic<bool> stopping{false};
    std::condition_variable wake;
    std::mutex wait_mutex;
    std::thread worker;
    std::array<std::size_t, 12> indices{};
    std::array<double, 12> factors{};
    std::array<std::string, 12> raw_names{};
    std::string id;
    int socket_fd = -1;
    sockaddr_in destination{};

    Impl(int port, const std::filesystem::path& mapping) : id(session_id()) {
        if (port < 1 || port > 65535) throw std::invalid_argument("invalid telemetry port");
        std::ifstream input(mapping);
        if (!input) throw std::runtime_error("cannot open telemetry glove mapping");
        const auto config = Json::parse(input);
        const auto& motors = config.at("motors");
        if (!motors.is_object() || motors.size() != 12) throw std::runtime_error("telemetry mapping requires 12 motors");
        std::array<bool, 12> used{};
        for (std::size_t raw = 0; raw < 12; ++raw) {
            const auto& motor = motors.at(std::to_string(raw + 1));
            const auto name = motor.at("joint").get<std::string>();
            const auto it = std::find_if(glove_names.begin(), glove_names.end(), [&](const char* n) { return name == n; });
            if (it == glove_names.end()) throw std::runtime_error("unknown telemetry mapping joint: " + name);
            const auto index = static_cast<std::size_t>(it - glove_names.begin());
            if (used[index]) throw std::runtime_error("duplicate telemetry mapping joint: " + name);
            used[index] = true; indices[raw] = index; raw_names[raw] = name;
            const double sign = motor.value("sign", 1.0), scale = motor.value("scale", 1.0);
            factors[raw] = sign * scale;
            if (!std::isfinite(sign) || !std::isfinite(scale) || !std::isfinite(factors[raw]) || factors[raw] == 0)
                throw std::runtime_error("invalid telemetry mapping factor");
        }
        socket_fd = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (socket_fd < 0) throw std::runtime_error("cannot create telemetry UDP socket");
        destination.sin_family = AF_INET;
        destination.sin_port = htons(static_cast<std::uint16_t>(port));
        destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        try { worker = std::thread([this] { run(); }); }
        catch (...) { ::close(socket_fd); throw; }
    }
    ~Impl() {
        stopping.store(true, std::memory_order_release);
        wake.notify_one();
        worker.join();
        ::close(socket_fd);
    }
    void publish(const SessionTick& tick) noexcept {
        slots[back] = Sample{tick, ++sequence};
        const unsigned old = middle.exchange(back | 4U, std::memory_order_acq_rel);
        back = old & 3U;
        if (old & 4U) dropped.fetch_add(1, std::memory_order_relaxed);
        wake.notify_one();
    }
    Json packet(const Sample& sample) const {
        const auto& t = sample.tick;
        const bool hand_valid = t.hand.received_ns != 0;
        const bool glove_valid = t.glove.oldest_ns != 0;
        const bool force = t.force_commanded && t.glove_submit_ns != 0;
        // Inference can send a release torque without a corresponding contact
        // estimate; v1's contact schema requires a complete estimate.
        const bool linear = force && t.takeover && t.linear_force;
        const bool contact = force && t.takeover && !t.linear_force;
        const bool position = !t.takeover && !force && t.glove_submit_ns != 0;
        const auto published = now_ns();
        Json raw_pos = Json::array(), raw_current = Json::array(), raw_stamp = Json::array(), raw_target = Json::array();
        if (contact && t.force_geometry_valid && t.contact_phase == ContactPhase::not_configured)
            throw std::runtime_error("missing contact phase");
        if (contact) for (double value : t.virtual_hand_nm)
            if (!std::isfinite(value)) throw std::runtime_error("invalid virtual torque estimate");
        for (std::size_t i = 0; i < 12; ++i) {
            const auto idx = indices[i];
            const bool valid = t.glove.received_ns[idx] != 0;
            raw_pos.push_back(valid ? number(t.glove.raw_degrees[idx]) : Json(nullptr));
            raw_current.push_back(valid ? number(t.glove.measured_current_a[idx]) : Json(nullptr));
            raw_stamp.push_back(stamp(t.glove.received_ns[idx]));
            raw_target.push_back(force ? number(t.glove_target[idx] * factors[i]) : Json(nullptr));
        }
        Json errors = Json::array();
        for (std::size_t i = 0; i < 20; ++i)
            errors.push_back(hand_valid && t.wuji_submit_ns ? number(t.wuji_target[i] - t.hand.position_rad[i]) : Json(nullptr));
        const auto state = !contact ? "not_configured" : !t.force_geometry_valid ? "waiting_geometry" : phase(t.contact_phase);
        Json obj = {
            {"schema_version", 1}, {"session_id", id}, {"sequence", sample.sequence}, {"source", "hardware"},
            {"timestamp_ns", std::max(t.hand.received_ns, t.glove.oldest_ns)},
            {"wuji_received_ns", t.hand.received_ns}, {"glove_received_ns", t.glove.oldest_ns},
            {"published_ns", published}, {"cycle_started_ns", stamp(t.cycle_started_ns)},
            {"cycle_completed_ns", stamp(t.cycle_completed_ns)}, {"wuji_submit_ns", stamp(t.wuji_submit_ns)},
            {"glove_submit_ns", stamp(t.glove_submit_ns)}, {"executed", t.wuji_submit_ns != 0},
            {"glove_output_source_ns", stamp(t.glove_output_source_ns)},
            {"mode", t.takeover ? "glove-to-hand2" : "inference-to-both"},
            {"mapping_mode", contact ? "contact_impedance_estimate" : linear ? "max_torque_linear_estimate" : "disabled"},
            {"feedback_state", (contact || linear) && t.force_geometry_valid ? "active" : "disabled"},
            {"jacobian_age_ns", (contact || linear) && t.jacobian_source_ns && published >= t.jacobian_source_ns ? Json(published - t.jacobian_source_ns) : Json(nullptr)},
            {"contact_state", state}, {"contact_active_mask", contact ? Json(t.contact) : Json(nullptr)},
            {"virtual_hand_torque_nm", contact ? values(t.virtual_hand_nm) : Json(nullptr)},
            {"contact_reflection_gain", nullptr}, {"glove_projected_torque_nm", values(Vec12{}, false)},
            {"wuji_joint_names", wuji_names}, {"glove_joint_names", raw_names},
            {"wuji_position_rad", hand_valid ? values(t.hand.position_rad) : Json(nullptr)},
            {"glove_position_rad", glove_valid ? values(t.glove_pose.joint_rad) : Json(nullptr)},
            {"glove_position_names", glove_names}, {"glove_raw_position_deg", raw_pos}, {"glove_motor_received_ns", raw_stamp},
            {"wuji_target_rad", t.wuji_submit_ns ? values(t.wuji_target) : Json(nullptr)},
            {"glove_target_position_rad", nullptr},
            {"glove_target_motor_rad", position ? values(t.glove_target) : Json(nullptr)},
            {"wuji_current_a", values(t.hand.effort_a, hand_valid && t.hand.has_effort)},
            {"wuji_velocity_rad_s", values(t.hand.velocity_rad_s, hand_valid && t.hand.has_velocity)},
            {"wuji_tracking_error_rad", errors}, {"wuji_load_fraction", values(t.wuji_load_fraction, linear && t.force_geometry_valid)},
            {"glove_current_a", raw_current}, {"glove_target", raw_target}, {"glove_target_unit", force ? Json("Nm") : Json(nullptr)}
        };
        return obj;
    }
    void run() noexcept {
        for (;;) {
            if (middle.load(std::memory_order_acquire) & 4U) {
                const unsigned old = middle.exchange(front, std::memory_order_acq_rel);
                front = old & 3U;
                try {
                    const auto bytes = packet(slots[front]).dump();
                    if (bytes.size() > 16384 || ::sendto(socket_fd, bytes.data(), bytes.size(), MSG_DONTWAIT,
                        reinterpret_cast<const sockaddr*>(&destination), sizeof(destination)) != static_cast<ssize_t>(bytes.size()))
                        dropped.fetch_add(1, std::memory_order_relaxed);
                } catch (...) { dropped.fetch_add(1, std::memory_order_relaxed); }
            } else if (stopping.load(std::memory_order_acquire)) break;
            else { std::unique_lock<std::mutex> lock(wait_mutex); wake.wait_for(lock, std::chrono::milliseconds(10)); }
        }
    }
};
TelemetryPublisher::TelemetryPublisher(int port, const std::filesystem::path& mapping)
    : impl_(std::make_unique<Impl>(port, mapping)) {}
TelemetryPublisher::~TelemetryPublisher() = default;
void TelemetryPublisher::publish(const SessionTick& tick) noexcept { impl_->publish(tick); }
std::uint64_t TelemetryPublisher::dropped_packets() const noexcept { return impl_->dropped.load(std::memory_order_relaxed); }
} // namespace motorhand::native
