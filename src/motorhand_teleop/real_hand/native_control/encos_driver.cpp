#include "encos_driver.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace motorhand::native {
namespace {
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr double half_lsb = 10.0 / 4095.0;
constexpr std::uint64_t feedback_age = 100000000;
constexpr std::array<const char*, 12> names = {
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_pip", "middle_mcp_flex", "middle_pip",
    "ring_mcp_flex", "ring_pip", "pinky_mcp_flex", "pinky_pip"};
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
json load_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open Encos configuration: " + path);
    return json::parse(in);
}
std::string text_file(const std::filesystem::path& p) {
    std::ifstream f(p); std::string s; f >> s; return s;
}
std::vector<std::string> discover() {
    std::vector<std::string> ports;
    for (const auto& e : std::filesystem::directory_iterator("/sys/class/tty")) {
        std::error_code ec;
        auto p = std::filesystem::canonical(e.path() / "device", ec);
        if (ec) continue;
        while (!p.empty() && p != p.root_path()) {
            if (text_file(p / "idVendor") == "16d0" && text_file(p / "idProduct") == "117e") {
                ports.push_back("/dev/" + e.path().filename().string()); break;
            }
            p = p.parent_path();
        }
    }
    std::sort(ports.begin(), ports.end());
    return ports;
}
std::array<unsigned char, 8> word_bytes(std::uint64_t w) {
    std::array<unsigned char, 8> b{};
    for (int i = 7; i >= 0; --i) { b[i] = static_cast<unsigned char>(w); w >>= 8; }
    return b;
}
std::array<unsigned char, 8> servo(double deg, double rpm, double amps) {
    float f = static_cast<float>(deg); std::uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
    auto speed = static_cast<std::uint64_t>(std::lround(std::clamp(rpm, 0.0, 3276.7) * 10));
    auto current = static_cast<std::uint64_t>(std::lround(std::clamp(amps, 0.0, 409.5) * 10));
    return word_bytes((1ULL << 61) | (std::uint64_t(bits) << 29) | (speed << 14) | (current << 2) | 2);
}
std::array<unsigned char, 8> mit(unsigned pos, double kp, double kd, unsigned torque) {
    auto k = static_cast<std::uint64_t>(std::lround(kp / 500.0 * 4095));
    auto d = static_cast<std::uint64_t>(std::lround(kd / 5.0 * 511));
    // Python's round uses ties-to-even: zero velocity encodes as 2048.
    return word_bytes((k << 49) | (d << 40) | (std::uint64_t(pos) << 24) | (2048ULL << 12) | torque);
}
std::string frame(unsigned id, const unsigned char* bytes, unsigned size) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out; out.reserve(6 + size * 2); out += 't';
    out += hex[(id >> 8) & 15]; out += hex[(id >> 4) & 15]; out += hex[id & 15];
    out += hex[size];
    for (unsigned i = 0; i < size; ++i) { out += hex[bytes[i] >> 4]; out += hex[bytes[i] & 15]; }
    out += '\r'; return out;
}
int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}
struct CanFrame { unsigned id = 0, size = 0; std::array<unsigned char, 8> data{}; };
bool parse(const std::string& s, CanFrame& f) {
    if (s.size() < 5 || s[0] != 't') return false;
    int a = hex_digit(s[1]), b = hex_digit(s[2]), c = hex_digit(s[3]), n = hex_digit(s[4]);
    if (a < 0 || b < 0 || c < 0 || n < 0 || n > 8 || s.size() < 5 + 2 * unsigned(n)) return false;
    f.id = (a << 8) | (b << 4) | c; f.size = n;
    for (int i = 0; i < n; ++i) {
        int hi = hex_digit(s[5 + 2*i]), lo = hex_digit(s[6 + 2*i]);
        if (hi < 0 || lo < 0) return false;
        f.data[i] = (hi << 4) | lo;
    }
    return true;
}
void send_text(int fd, const std::string& s) {
    // Nonblocking writes must be atomic: a partial SLCAN record is a failed channel.
    auto n = ::write(fd, s.data(), s.size());
    if (n != static_cast<ssize_t>(s.size())) throw std::runtime_error("Encos SLCAN serial write failed");
}
int open_port(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("Cannot open Encos serial port " + path);
    try {
        termios t{};
        if (tcgetattr(fd, &t) != 0) throw std::runtime_error("Encos tcgetattr failed");
        cfmakeraw(&t); cfsetispeed(&t, B115200); cfsetospeed(&t, B115200);
        t.c_cflag |= CLOCAL | CREAD; t.c_cflag &= ~CRTSCTS;
        if (tcsetattr(fd, TCSANOW, &t) != 0) throw std::runtime_error("Encos tcsetattr failed");
        tcflush(fd, TCIOFLUSH);
        send_text(fd, "C\rS8\rO\r");
        return fd;
    } catch (...) { ::close(fd); throw; }
}
}

struct EncosDriver::Impl {
    struct Motor { int index{}; double zero{}, sign{}, scale{}, factor{}, min{}, max{}; };
    struct Feedback { double angle{}, current{}; std::uint64_t stamp{}; int error{}; };
    struct Command {
        std::array<unsigned char, 8> data{};
        std::uint64_t deadline{};
        bool enabled{};
        std::uint64_t generation{};
        bool stop{};
    };
    struct Channel {
        int fd = -1; std::vector<int> ids; std::thread thread; std::string input;
        std::array<std::array<unsigned char, 8>, 12> sent{};
        std::array<bool, 12> has_sent{};
    };
    std::array<Motor, 12> motors{};
    std::array<double, 12> force_limits{};
    std::array<double, 12> home_current{};
    bool has_force = false, calibrated = false, connected = false, homed = false, force = false;
    std::array<Feedback, 12> feedback{};
    std::array<Command, 12> commands{};
    std::array<std::uint64_t, 12> submitted_generation{}, submitted_ns{};
    std::uint64_t generation{};
    bool output_latched{}, submitting{}, owned{}, stopping{};
    std::array<Channel, 3> channels;
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
    std::vector<std::string> offline_ports;
#endif
    mutable std::mutex mutex;
    std::condition_variable command_changed;
    std::atomic<bool> running{false};
    std::string channel_error;

    Impl(const std::string& mapping_path, const std::string& limits_path) {
        auto cfg = load_json(mapping_path);
        calibrated = cfg.value("calibrated", false) &&
                     cfg.value("id_order_confirmed", false) &&
                     cfg.value("direction_confirmed", false);
        // motor_hand.py's legacy fixed rule applies when neither setting exists;
        // explicit settings must also respect encos_home.py's fixed safety rule.
        double default_current = cfg.value("home_current_a", 1.0);
        json overrides = cfg.value("home_current_a_by_motor", json::object());
        if (!overrides.is_object()) throw std::runtime_error("Encos homing current overrides must be an object");
        if (!cfg.contains("home_current_a") && !cfg.contains("home_current_a_by_motor"))
            overrides["1"] = 1.5;
        if (!std::isfinite(default_current) || default_current <= 0.0 || default_current > 409.5)
            throw std::runtime_error("Invalid Encos default homing current");
        for (auto it = overrides.begin(); it != overrides.end(); ++it) {
            bool known = false;
            for (int id = 1; id <= 12; ++id) if (it.key() == std::to_string(id)) known = true;
            if (!known || !it.value().is_number())
                throw std::runtime_error("Invalid Encos homing current override");
        }
        for (int i = 0; i < 12; ++i) {
            const auto key = std::to_string(i + 1);
            double current = overrides.contains(key) ? overrides.at(key).get<double>() : default_current;
            double expected = i == 0 ? 1.5 : 1.0;
            if (!std::isfinite(current) || current <= 0.0 || current > 409.5 ||
                std::abs(current - expected) > 1e-9)
                throw std::runtime_error("Encos homing current violates fixed J1/other-axis safety rule");
            home_current[i] = current;
        }
        const auto& m = cfg.at("motors");
        if (!m.is_object() || m.size() != 12) throw std::runtime_error("Encos mapping must have 12 motors");
        std::array<bool, 12> used{};
        for (int id = 1; id <= 12; ++id) {
            const auto& v = m.at(std::to_string(id));
            std::string joint = v.at("joint").get<std::string>();
            auto it = std::find_if(names.begin(), names.end(), [&](const char* n) { return joint == n; });
            if (it == names.end()) throw std::runtime_error("Unknown Encos joint " + joint);
            auto idx = static_cast<int>(it - names.begin());
            if (used[idx]) throw std::runtime_error("Duplicate Encos joint");
            used[idx] = true;
            auto& x = motors[id-1]; x.index = idx;
            x.zero = v.value("zero_deg", 0.0);
            double sign = v.value("sign", 1.0), scale = v.value("scale", 1.0);
            x.sign = sign; x.scale = scale; x.factor = sign * scale;
            x.min = v.value("min_deg", -360.0); x.max = v.value("max_deg", 360.0);
            double speed = v.value("speed_rpm", 30.0), current = v.value("current_limit_a", 0.8);
            if (!std::isfinite(x.zero) || !std::isfinite(x.factor) || x.factor == 0 ||
                !std::isfinite(x.min) || !std::isfinite(x.max) || x.min >= x.max ||
                !std::isfinite(speed) || speed < 0 || !std::isfinite(current) || current < 0)
                throw std::runtime_error("Invalid Encos motor calibration");
        }
        if (!limits_path.empty()) {
            auto limits = load_json(limits_path);
            if (limits.is_object() && limits.contains("glove_max_output_torque_nm")) limits = limits.at("glove_max_output_torque_nm");
            else if (limits.is_object() && limits.contains("force_torque_limits_nm")) limits = limits.at("force_torque_limits_nm");
            if ((!limits.is_array() && !limits.is_object()) || limits.size() != 12)
                throw std::runtime_error("Encos force limits must cover 12 raw axes");
            for (int i = 0; i < 12; ++i) {
                double v = limits.is_array() ? limits.at(i).get<double>() : limits.at(std::to_string(i+1)).get<double>();
                if (!std::isfinite(v) || v <= half_lsb || v > 10.0) throw std::runtime_error("Invalid Encos force limit");
                force_limits[i] = v;
            }
            has_force = true;
        }
    }
    void require_connected() const { if (!connected) throw std::runtime_error("Encos not connected"); }
    void require_calibrated() const { if (!calibrated) throw std::runtime_error("Encos mapping is not calibrated"); }
    void require_unlatched() const {
        if (output_latched) throw std::runtime_error("Encos output fault latched; reconnect required");
        if (stopping) throw std::runtime_error("Encos stop in progress");
    }
    void check_deadline(std::uint64_t deadline) {
        if (now_ns() >= deadline) {
            for (auto& cmd : commands) cmd.enabled = false;
            force = false; output_latched = true; command_changed.notify_all();
            throw std::runtime_error("Encos command deadline expired; output fault latched");
        }
    }
    void submit_and_wait(std::array<Command, 12>& next, std::uint64_t deadline,
                         std::unique_lock<std::mutex>& lock) {
        check_deadline(deadline);
        if (submitting) throw std::runtime_error("Concurrent Encos frame submission");
        submitting = true;
        const auto serial = ++generation;
        for (auto& cmd : next) cmd.generation = serial;
        commands = next;
        auto complete = [&] {
            if (output_latched || stopping || !channel_error.empty() || !running.load()) return true;
            bool all_submitted = true;
            for (int i = 0; i < 12; ++i) {
                if (commands[i].generation != serial || !commands[i].enabled) return true;
                if (submitted_generation[i] != serial) all_submitted = false;
            }
            return all_submitted;
        };
        const auto until = Clock::time_point(std::chrono::nanoseconds(deadline));
        command_changed.wait_until(lock, until, complete);
        bool success = !output_latched && !stopping && channel_error.empty() && running.load();
        for (int i = 0; i < 12; ++i)
            success = success && commands[i].generation == serial && commands[i].enabled &&
                      submitted_generation[i] == serial && submitted_ns[i] < deadline;
        submitting = false;
        command_changed.notify_all();
        if (!success) {
            for (auto& cmd : commands) cmd.enabled = false;
            force = false;
            output_latched = true;
            command_changed.notify_all();
            throw std::runtime_error("Encos frame submission incomplete; all outputs cancelled and fault latched");
        }
    }
    GloveFrame frame_locked() const {
        require_connected();
        if (!channel_error.empty()) throw std::runtime_error(channel_error);
        GloveFrame out; out.oldest_ns = std::numeric_limits<std::uint64_t>::max();
        auto now = now_ns();
        for (int i = 0; i < 12; ++i) {
            const auto& f = feedback[i]; const auto& m = motors[i];
            if (!f.stamp || f.stamp > now || now - f.stamp > feedback_age || f.error || !std::isfinite(f.angle) || !std::isfinite(f.current))
                throw std::runtime_error("Encos motor " + std::to_string(i+1) + " stale, missing, invalid or faulted");
            out.raw_degrees[m.index] = f.angle;
            out.motor_rad[m.index] = (f.angle - m.zero) * m.factor * pi / 180.0;
            out.measured_current_a[m.index] = f.current;
            out.received_ns[m.index] = f.stamp;
            out.oldest_ns = std::min(out.oldest_ns, f.stamp);
        }
        return out;
    }
    void receive(Channel& ch, int timeout_ms, bool scan, std::vector<int>* found = nullptr) {
        pollfd p{ch.fd, POLLIN, 0};
        int result = ::poll(&p, 1, timeout_ms);
        if (result < 0) { if (errno == EINTR) return; throw std::runtime_error("Encos serial poll failed"); }
        if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) throw std::runtime_error("Encos serial disconnected");
        if (!(p.revents & POLLIN)) return;
        char buffer[512]; auto n = ::read(ch.fd, buffer, sizeof(buffer));
        if (n < 0) { if (errno == EAGAIN || errno == EINTR) return; throw std::runtime_error("Encos serial read failed"); }
        if (n == 0) return;
        auto stamp = now_ns();
        for (ssize_t k = 0; k < n; ++k) {
            char c = buffer[k];
            if (c != '\r' && c != '\n') { if (ch.input.size() < 128) ch.input += c; else ch.input.clear(); continue; }
            CanFrame f; bool valid = parse(ch.input, f); ch.input.clear();
            if (!valid) continue;
            if (scan && f.id == 0x7ff && f.size >= 5 && f.data[0] == 0xff && f.data[1] == 0xff && f.data[2] == 1) {
                int id = (f.data[3] << 8) | f.data[4];
                if (id >= 1 && id <= 12 && std::find(found->begin(), found->end(), id) == found->end()) found->push_back(id);
                continue;
            }
            if (scan || f.id < 1 || f.id > 12 || f.size != 8 || std::find(ch.ids.begin(), ch.ids.end(), int(f.id)) == ch.ids.end()) continue;
            int i = f.id - 1;
            if (ch.has_sent[i] && f.data == ch.sent[i]) continue;
            int type = f.data[0] >> 5; Feedback fb; fb.error = f.data[0] & 31; fb.stamp = stamp;
            if (type == 2) {
                std::uint32_t bits = (std::uint32_t(f.data[1]) << 24) | (std::uint32_t(f.data[2]) << 16) | (std::uint32_t(f.data[3]) << 8) | f.data[4];
                float angle; std::memcpy(&angle, &bits, sizeof(angle)); fb.angle = angle;
                int raw = (f.data[5] << 8) | f.data[6]; if (raw >= 32768) raw -= 65536;
                fb.current = raw / 100.0;
            } else if (type == 1) {
                unsigned pos = (f.data[1] << 8) | f.data[2];
                fb.angle = (-5.0 + pos * 10.0 / 65535.0) * 180.0 / pi;
                fb.current = -2.0 + (((f.data[4] & 15) << 8) | f.data[5]) * 4.0 / 4095.0;
            } else continue;
            std::lock_guard<std::mutex> lock(mutex); feedback[i] = fb; command_changed.notify_all();
        }
    }
    void run(Channel& ch) {
        try {
            std::size_t slot = 0;
            while (running.load()) {
                auto end = Clock::now() + std::chrono::milliseconds(5);
                int id = ch.ids[slot]; slot = (slot + 1) % ch.ids.size();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto& cmd = commands[id-1];
                    bool active = cmd.enabled && now_ns() < cmd.deadline &&
                                  (!output_latched || cmd.stop);
                    // Normal read-only operation uses the SDK's zero-current
                    // servo polling command, not passive observation.
                    if (active || !stopping) {
                        auto bytes = active ? cmd.data : servo(feedback[id-1].angle, 0, 0);
                        auto packet = frame(id, bytes.data(), 8);
                        if (active && now_ns() >= cmd.deadline) {
                            active = false;
                            bytes = servo(feedback[id-1].angle, 0, 0);
                            packet = frame(id, bytes.data(), 8);
                        }
                        if (active || !stopping) {
                            send_text(ch.fd, packet);
                            auto written = now_ns();
                            ch.sent[id-1] = bytes; ch.has_sent[id-1] = true;
                            if (active && cmd.generation && submitted_generation[id-1] != cmd.generation) {
                                submitted_generation[id-1] = cmd.generation;
                                submitted_ns[id-1] = written;
                                command_changed.notify_all();
                            }
                        }
                    }
                }
                while (running.load() && Clock::now() < end) {
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
                    receive(ch, static_cast<int>(std::max<std::int64_t>(0, ms)), false);
                    if (ms <= 0) std::this_thread::sleep_until(end);
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex); channel_error = e.what();
            for (auto& cmd : commands) if (!cmd.stop) cmd.enabled = false;
            output_latched = true; force = false; command_changed.notify_all();
        }
    }
};

EncosDriver::EncosDriver(const std::string& mapping_path, const std::string& force_limits_path)
    : impl_(std::make_unique<Impl>(mapping_path, force_limits_path)) {}
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
EncosDriver::EncosDriver(const std::string& mapping_path, const std::string& force_limits_path,
                         const OfflineTestPorts& ports)
    : impl_(std::make_unique<Impl>(mapping_path, force_limits_path)) {
    for (const auto& path : ports.paths) {
        const std::string prefix = "/dev/pts/";
        if (path.compare(0, prefix.size(), prefix) != 0 || path.size() == prefix.size() ||
            !std::all_of(path.begin() + prefix.size(), path.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
            std::filesystem::canonical(path).string() != path)
            throw std::invalid_argument("Offline Encos tests require canonical /dev/pts/N paths");
        if (std::find(impl_->offline_ports.begin(), impl_->offline_ports.end(), path) != impl_->offline_ports.end())
            throw std::invalid_argument("Offline Encos PTY paths must be distinct");
        impl_->offline_ports.push_back(path);
    }
}
#endif

EncosDriver::~EncosDriver() {
    try { close(); } catch (...) {}
}

void EncosDriver::connect() {
    auto& d = *impl_;
    if (d.connected || d.running.load()) throw std::runtime_error("Encos already connected");
    std::vector<std::string> ports;
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
    if (!d.offline_ports.empty()) ports = d.offline_ports;
    else
#endif
    ports = discover();
    if (ports.size() != 3) throw std::runtime_error("Expected exactly three CANable2 USB-CAN adapters");
    try {
        std::array<bool, 12> seen{};
        for (int i = 0; i < 3; ++i) {
            auto& ch = d.channels[i];
            ch.fd = open_port(ports[i]); ch.input.clear(); ch.ids.clear(); ch.has_sent.fill(false);
            const unsigned char probe[] = {0xff, 0xff, 0x00, 0x82};
            send_text(ch.fd, frame(0x7ff, probe, 4));
            auto end = Clock::now() + std::chrono::milliseconds(150);
            while (Clock::now() < end) d.receive(ch, 5, true, &ch.ids);
            std::sort(ch.ids.begin(), ch.ids.end());
            if (ch.ids.empty()) throw std::runtime_error("Encos USB-CAN channel has no motors");
            for (int id : ch.ids) {
                if (seen[id-1]) throw std::runtime_error("Duplicate Encos motor ID across channels");
                seen[id-1] = true;
            }
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::runtime_error("Encos scan did not find all 12 motors");
        {
            std::lock_guard<std::mutex> lock(d.mutex);
            d.channel_error.clear(); d.feedback = {}; d.commands = {};
            d.output_latched = false; d.submitting = false; d.owned = false; d.stopping = false;
            d.submitted_generation = {}; d.submitted_ns = {};
            d.connected = true; d.homed = false; d.force = false;
        }
        d.running.store(true);
        for (auto& ch : d.channels) ch.thread = std::thread([&d, &ch] { d.run(ch); });
        auto end = Clock::now() + std::chrono::seconds(1);
        while (true) {
            try { read(); break; }
            catch (const std::exception&) {
                if (Clock::now() >= end) throw;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    } catch (...) { close(); throw; }
}

GloveFrame EncosDriver::read() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->frame_locked();
}
std::array<EncosMotorCalibration, 12> EncosDriver::calibration() const {
    const auto& d = *impl_;
    std::array<EncosMotorCalibration, 12> result{};
    for (int i = 0; i < 12; ++i) {
        const auto& m = d.motors[i];
        double a = (m.min - m.zero) * m.factor * pi / 180.0;
        double b = (m.max - m.zero) * m.factor * pi / 180.0;
        result[m.index] = {i + 1, m.zero, m.sign, m.scale, m.min, m.max,
                           std::min(a, b), std::max(a, b), d.force_limits[i], d.has_force};
    }
    return result;
}

std::array<std::uint64_t, 12> EncosDriver::last_command_write_ns() const {
    const auto& d = *impl_;
    std::lock_guard<std::mutex> lock(d.mutex);
    std::array<std::uint64_t, 12> result{};
    for (int i = 0; i < 12; ++i) result[d.motors[i].index] = d.submitted_ns[i];
    return result;
}


void EncosDriver::home(double tolerance_rad, double timeout_s) {
    if (!std::isfinite(tolerance_rad) || tolerance_rad <= 0 || !std::isfinite(timeout_s) || timeout_s <= 0 ||
        timeout_s > 86400) throw std::invalid_argument("Invalid Encos homing tolerance or timeout");
    auto& d = *impl_;
    auto start = now_ns();
    auto finish = start + static_cast<std::uint64_t>(timeout_s * 1e9);
    try {
        {
            std::lock_guard<std::mutex> lock(d.mutex);
            d.require_calibrated(); d.require_unlatched(); d.frame_locked();
            if (d.submitting) throw std::runtime_error("Encos frame submission in progress");
            for (const auto& m : d.motors)
                if (m.zero < m.min || m.zero > m.max) throw std::runtime_error("Encos zero outside soft limits");
            d.homed = false; d.force = false;
            d.owned = true;
            for (int i = 0; i < 12; ++i) {
                auto& c = d.commands[i]; auto& m = d.motors[i]; auto angle = d.feedback[i].angle;
                c.enabled = angle >= m.min && angle <= m.max;
                c.generation = 0; c.stop = false;
                c.data = servo(angle, 0, d.home_current[i]);
                c.deadline = std::min(finish, now_ns() + std::uint64_t{250000000});
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto motion_start = now_ns();
        while (true) {
            {
                std::lock_guard<std::mutex> lock(d.mutex);
                d.require_unlatched();
                if (!d.owned) throw std::runtime_error("Encos homing output was released");
                if (now_ns() >= finish) throw std::runtime_error("Encos homing timeout");
                auto f = d.frame_locked();
                bool arrived = f.oldest_ns >= motion_start;
                for (int i = 0; i < 12; ++i) {
                    const auto& m = d.motors[i];
                    if (std::abs(d.feedback[i].angle - m.zero) * pi / 180.0 > tolerance_rad) arrived = false;
                    auto& c = d.commands[i];
                    c.data = servo(m.zero, 5.0, d.home_current[i]);
                    c.deadline = std::min(finish, now_ns() + std::uint64_t{250000000}); c.enabled = true;
                }
                if (arrived) { d.homed = true; return; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    } catch (...) { disable(); throw; }
}

void EncosDriver::enable_force() {
    auto& d = *impl_;
    std::lock_guard<std::mutex> lock(d.mutex);
    d.require_calibrated(); d.require_unlatched(); d.frame_locked();
    if (d.submitting) throw std::runtime_error("Encos frame submission in progress");
    if (!d.homed) throw std::runtime_error("Encos force control requires homing");
    if (!d.has_force) throw std::runtime_error("Encos force control requires explicit torque limits");
    // Do not manufacture a command deadline: the caller's first torque frame takes ownership.
    for (auto& c : d.commands) c.enabled = false;
    d.force = true;
}

Vec12 EncosDriver::write_position(const Vec12& position, std::uint64_t deadline_ns) {
    auto& d = *impl_; std::unique_lock<std::mutex> lock(d.mutex);
    d.require_calibrated(); d.require_unlatched(); d.frame_locked(); d.check_deadline(deadline_ns);
    if (!d.homed) throw std::runtime_error("Encos position control requires homing");
    std::array<Impl::Command, 12> next{}; Vec12 actual{};
    for (int i = 0; i < 12; ++i) {
        const auto& m = d.motors[i]; double v = position[m.index];
        if (!std::isfinite(v)) throw std::invalid_argument("Non-finite Encos position");
        double low = std::max(-5.0, m.min * pi / 180.0), high = std::min(5.0, m.max * pi / 180.0);
        int low_code = static_cast<int>(std::ceil((low + 5.0) * 65535.0 / 10.0));
        int high_code = static_cast<int>(std::floor((high + 5.0) * 65535.0 / 10.0));
        if (low_code > high_code) throw std::runtime_error("Encos soft limits outside MIT position range");
        double raw = m.zero * pi / 180.0 + v / m.factor;
        int code = static_cast<int>(std::lround((std::clamp(raw, low, high) + 5.0) * 65535.0 / 10.0));
        code = std::clamp(code, low_code, high_code);
        double commanded = -5.0 + code * 10.0 / 65535.0;
        actual[m.index] = (commanded - m.zero * pi / 180.0) * m.factor;
        next[i] = {mit(code, 45.0, 0.5, 2048), deadline_ns, true};
    }
    d.force = false; d.owned = true; d.submit_and_wait(next, deadline_ns, lock);
    return actual;
}

Vec12 EncosDriver::write_torque(const Vec12& torque, std::uint64_t deadline_ns) {
    auto& d = *impl_; std::unique_lock<std::mutex> lock(d.mutex);
    d.require_calibrated(); d.require_unlatched(); d.frame_locked(); d.check_deadline(deadline_ns);
    if (!d.homed || !d.force || !d.has_force) throw std::runtime_error("Encos force mode not enabled");
    std::array<Impl::Command, 12> next{}; Vec12 actual{};
    for (int i = 0; i < 12; ++i) {
        const auto& m = d.motors[i]; double v = torque[m.index];
        if (!std::isfinite(v)) throw std::invalid_argument("Non-finite Encos torque");
        double cap = d.force_limits[i] - half_lsb;
        double raw = std::clamp(v * m.factor, -cap, cap);
        if (d.feedback[i].angle <= m.min) raw = std::max(raw, half_lsb);
        else if (d.feedback[i].angle >= m.max) raw = std::min(raw, -half_lsb);
        // Resolve zero's half-LSB tie explicitly, especially at the lower soft limit.
        int code = static_cast<int>(std::lround((raw + 10.0) * 4095.0 / 20.0));
        if (d.feedback[i].angle <= m.min) code = std::max(code, 2048);
        if (d.feedback[i].angle >= m.max) code = std::min(code, 2047);
        code = std::clamp(code, 0, 4095);
        while (code > 0 && -10.0 + code * 20.0 / 4095.0 > d.force_limits[i]) --code;
        while (code < 4095 && -10.0 + code * 20.0 / 4095.0 < -d.force_limits[i]) ++code;
        actual[m.index] = (-10.0 + code * 20.0 / 4095.0) / m.factor;
        next[i] = {mit(32768, 0.0, 0.0, code), deadline_ns, true};
    }
    d.owned = true; d.submit_and_wait(next, deadline_ns, lock);
    return actual;
}

void EncosDriver::disable() {
    auto& d = *impl_;
    std::unique_lock<std::mutex> lock(d.mutex);
    if (d.stopping) throw std::runtime_error("Encos stop already in progress");
    d.stopping = true;
    for (auto& cmd : d.commands) cmd.enabled = false;
    d.force = false;
    d.command_changed.notify_all();
    const auto deadline = now_ns() + std::uint64_t{100000000};
    const auto until = Clock::time_point(std::chrono::nanoseconds(deadline));
    if (!d.command_changed.wait_until(lock, until, [&] { return !d.submitting; })) {
        d.output_latched = true; d.stopping = false;
        throw std::runtime_error("Encos stop could not cancel in-flight frame before deadline");
    }
    if (!d.owned) {
        d.stopping = false;
        if (!d.channel_error.empty()) throw std::runtime_error("Encos channel failure: " + d.channel_error);
        return;
    }
    if (!d.connected || !d.running.load()) {
        d.output_latched = true; d.stopping = false;
        throw std::runtime_error("Encos stop failed: channel workers are not running");
    }
    const auto serial = ++d.generation;
    for (int i = 0; i < 12; ++i) {
        double angle = std::isfinite(d.feedback[i].angle) ? d.feedback[i].angle : 0.0;
        d.commands[i] = {servo(angle, 0, 0), deadline, true, serial, true};
    }
    d.command_changed.wait_until(lock, until, [&] {
        for (int i = 0; i < 12; ++i)
            if (d.submitted_generation[i] != serial) return false;
        return true;
    });
    bool complete = d.channel_error.empty();
    for (int i = 0; i < 12; ++i)
        complete = complete && d.submitted_generation[i] == serial && d.submitted_ns[i] < deadline;
    for (auto& cmd : d.commands) cmd.enabled = false;
    d.stopping = false;
    if (!complete) {
        d.output_latched = true;
        throw std::runtime_error("Encos stop incomplete: zero-current serial submission failed or expired" +
                                 (d.channel_error.empty() ? std::string{} : ": " + d.channel_error));
    }
    d.owned = false;
}

void EncosDriver::close() {
    auto& d = *impl_;
    std::string errors;
    try { disable(); } catch (const std::exception& e) { errors = e.what(); }
    d.running.store(false);
    d.command_changed.notify_all();
    for (auto& ch : d.channels) {
        if (ch.thread.joinable()) ch.thread.join();
        if (ch.fd >= 0) {
            try { send_text(ch.fd, "C\r"); }
            catch (const std::exception& e) {
                if (!errors.empty()) errors += "; ";
                errors += std::string("Encos SLCAN close command failed: ") + e.what();
            }
            if (::close(ch.fd) != 0) {
                if (!errors.empty()) errors += "; ";
                errors += "Encos serial close failed";
            }
            ch.fd = -1;
        }
        ch.ids.clear();
    }
    {
        std::lock_guard<std::mutex> lock(d.mutex);
        d.connected = false; d.homed = false; d.force = false;
        d.feedback = {}; d.commands = {};
    }
    if (!errors.empty()) throw std::runtime_error(errors);
}

}  // namespace motorhand::native
