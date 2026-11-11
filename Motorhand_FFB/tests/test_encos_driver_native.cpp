// Offline-only integration test. Compile this file and encos_driver.cpp with
// -DMOTORHAND_ENCOS_OFFLINE_TEST; all serial devices are private pseudoterminals.
#include "motorhand_teleop/real_hand/native_control/encos_driver.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <unistd.h>

using namespace motorhand::native;
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr const char* names[] = {"thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip", "index_mcp_flex", "index_pip", "middle_mcp_flex", "middle_pip", "ring_mcp_flex", "ring_pip", "pinky_mcp_flex", "pinky_pip"};
std::uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
bool near(double a, double b, double tol = 1e-8) { return std::abs(a - b) <= tol; }
template<class F> void throws(F&& f, const char* message) {
    bool failed = false; try { f(); } catch (const std::exception&) { failed = true; }
    check(failed, message);
}

struct Files {
    std::filesystem::path dir, mapping, limits;
    Files() {
        char path[] = "/tmp/encos-native-test-XXXXXX";
        char* result = mkdtemp(path); check(result, "mkdtemp failed"); dir = result;
        mapping = dir / "mapping.json"; limits = dir / "limits.json";
        json config;
        config["calibrated"] = true;
        config["id_order_confirmed"] = true;
        config["direction_confirmed"] = true;
        config["home_current_a"] = 1.0;
        config["home_current_a_by_motor"] = {{"1", 1.5}};
        for (int id = 1; id <= 12; ++id) {
            config["motors"][std::to_string(id)] = {
                {"joint", names[12-id]}, {"zero_deg", 0.0},
                {"sign", id % 2 ? 1.0 : -1.0}, {"scale", 1.5},
                {"min_deg", -30.0}, {"max_deg", 30.0}};
        }
        std::ofstream(mapping) << config.dump();
        json force; force["glove_max_output_torque_nm"] = std::vector<double>(12, 1.0);
        std::ofstream(limits) << force.dump();
    }
    ~Files() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
};
int digit(char c) {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'A' && c <= 'F') return c-'A'+10;
    if (c >= 'a' && c <= 'f') return c-'a'+10;
    return -1;
}
struct Frame { int id{}; std::vector<unsigned char> bytes; std::uint64_t received{}; };
bool parse_frame(const std::string& s, Frame& out) {
    if (s.size() < 5 || s[0] != 't') return false;
    int a = digit(s[1]), b = digit(s[2]), c = digit(s[3]), len = digit(s[4]);
    if (a < 0 || b < 0 || c < 0 || len < 0 || len > 8 || s.size() != 5 + std::size_t(len)*2) return false;
    out.id = (a<<8)|(b<<4)|c; out.bytes.clear(); out.received = now_ns();
    for (int i=0; i<len; ++i) {
        int hi = digit(s[5+i*2]), lo = digit(s[6+i*2]);
        if (hi < 0 || lo < 0) return false;
        out.bytes.push_back((hi<<4)|lo);
    }
    return true;
}
std::string encode(int id, const std::vector<unsigned char>& data) {
    constexpr char hex[] = "0123456789ABCDEF";
    std::string s = "t";
    s += hex[(id>>8)&15]; s += hex[(id>>4)&15]; s += hex[id&15]; s += hex[data.size()];
    for (auto b : data) { s += hex[b>>4]; s += hex[b&15]; }
    s += '\r'; return s;
}
std::uint64_t word(const Frame& f) {
    check(f.bytes.size()==8, "expected eight-byte command");
    std::uint64_t result = 0; for (auto b : f.bytes) result = (result<<8)|b; return result;
}

class FakeBuses {
    struct Port {
        int fd = -1; std::string path, input; std::vector<int> ids; std::thread thread;
        std::atomic<bool> active{true};
    };
    std::array<Port, 3> ports_;
    std::atomic<bool> running_{true}, mit_feedback_{false};
    std::atomic<int> silent_id_{0}, fault_id_{0};
    std::mutex mutex_;
    std::condition_variable changed_;
    std::array<float, 12> angle_{};
    std::vector<Frame> received_;
    void send(int fd, int id, const std::vector<unsigned char>& bytes) {
        auto s = encode(id, bytes);
        // The slave can close immediately after its final command.
        (void)::write(fd, s.data(), s.size());
    }
    void on_frame(Port& p, const Frame& f) {
        if (f.id == 0x7ff && f.bytes == std::vector<unsigned char>{0xff,0xff,0,0x82}) {
            for (int id : p.ids) send(p.fd, 0x7ff, {0xff,0xff,1,static_cast<unsigned char>(id>>8),static_cast<unsigned char>(id)});
            return;
        }
        if (f.id < 1 || f.id > 12 || f.bytes.size() != 8 ||
            std::find(p.ids.begin(), p.ids.end(), f.id) == p.ids.end()) return;
        float angle;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            received_.push_back(f);
            auto w = word(f); int mode = w >> 61;
            if (mode == 1 && ((w >> 14) & 0x7fff) != 0 && ((w >> 2) & 0xfff) != 0) {
                std::uint32_t bits = (w >> 29) & 0xffffffffU;
                std::memcpy(&angle, &bits, sizeof(angle)); angle_[f.id-1] = angle;
            } else if (mode == 0 && ((w >> 49) & 0xfff) != 0) {
                unsigned pos = (w >> 24) & 0xffff;
                angle_[f.id-1] = static_cast<float>((-5.0 + pos * 10.0 / 65535.0) * 180.0 / pi);
            }
            angle = angle_[f.id-1]; changed_.notify_all();
        }
        if (silent_id_.load() == f.id || silent_id_.load() == -1) return;
        if (mit_feedback_.load()) {
            unsigned pos = static_cast<unsigned>(std::lround((std::clamp(double(angle)*pi/180.0, -5.0, 5.0)+5.0)*65535.0/10.0));
            constexpr unsigned current = 2048;
            send(p.fd, f.id, {0x20, static_cast<unsigned char>(pos>>8), static_cast<unsigned char>(pos),
                              0, static_cast<unsigned char>(current>>8), static_cast<unsigned char>(current), 100, 0});
            return;
        }
        std::uint32_t bits; std::memcpy(&bits, &angle, sizeof(bits));
        // Type-2 feedback carries big-endian IEEE754 degrees and signed centiamps.
        int current = f.id * 10;
        send(p.fd, f.id, {static_cast<unsigned char>(0x40 | (fault_id_.load() == f.id ? 1 : 0)), static_cast<unsigned char>(bits>>24), static_cast<unsigned char>(bits>>16),
             static_cast<unsigned char>(bits>>8), static_cast<unsigned char>(bits),
             static_cast<unsigned char>(current>>8), static_cast<unsigned char>(current), 100});
    }
    void run(Port& p) {
        while (running_.load() && p.active.load()) {
            pollfd fd{p.fd, POLLIN, 0}; int ready = poll(&fd, 1, 5);
            if (ready <= 0 || !(fd.revents & POLLIN)) continue;
            char data[512]; ssize_t n = ::read(p.fd, data, sizeof(data));
            if (n <= 0) continue; // PTY master may report EIO before the slave opens.
            for (ssize_t i=0; i<n; ++i) {
                if (data[i] == '\r' || data[i] == '\n') {
                    Frame f; if (parse_frame(p.input, f)) on_frame(p, f);
                    p.input.clear();
                } else if (p.input.size() < 128) p.input += data[i];
                else p.input.clear();
            }
        }
    }
public:
    explicit FakeBuses(const std::array<std::vector<int>, 3>& ids) {
        for (int i=0; i<3; ++i) {
            auto& p = ports_[i]; p.ids = ids[i];
            p.fd = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
            check(p.fd >= 0 && grantpt(p.fd) == 0 && unlockpt(p.fd) == 0, "creating PTY failed");
            char* slave = ptsname(p.fd); check(slave != nullptr, "ptsname failed"); p.path = slave;
        }
        for (auto& p : ports_) p.thread = std::thread([this, &p] { run(p); });
    }
    ~FakeBuses() {
        running_.store(false);
        for (auto& p : ports_) { if (p.thread.joinable()) p.thread.join(); if (p.fd >= 0) ::close(p.fd); }
    }
    void disconnect(int index) {
        auto& p = ports_.at(index);
        p.active.store(false);
        if (p.thread.joinable()) p.thread.join();
        if (p.fd >= 0) { ::close(p.fd); p.fd = -1; }
    }
    EncosDriver::OfflineTestPorts paths() const { return {{ports_[0].path, ports_[1].path, ports_[2].path}}; }
    void set_angle(int id, float degrees) { std::lock_guard<std::mutex> lock(mutex_); angle_[id-1] = degrees; }
    void silence(int id) { silent_id_.store(id); }
    void fault(int id) { fault_id_.store(id); }
    void mit_feedback(bool enabled) { mit_feedback_.store(enabled); }
    std::size_t mark() { std::lock_guard<std::mutex> lock(mutex_); return received_.size(); }
    std::vector<Frame> after(std::size_t mark, int kind, int wait_ms = 150) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait_for(lock, std::chrono::milliseconds(wait_ms), [&] {
            std::array<bool, 12> seen{};
            for (auto it = received_.begin()+mark; it != received_.end(); ++it) {
                auto w = word(*it); auto mode = w >> 61;
                bool match = kind == 0 ? mode == 0 && ((w >> 49) & 0xfff) != 0 :
                             kind == 1 ? mode == 1 && ((w >> 14) & 0x7fff) == 50 :
                             kind == 2 ? mode == 1 && ((w >> 14) & 0x7fff) == 0 && ((w >> 2) & 0xfff) == 0 :
                                         mode == 0 && ((w >> 49) & 0xfff) == 0 && ((w >> 40) & 0x1ff) == 0;
                if (match) seen[it->id-1] = true;
            }
            return std::all_of(seen.begin(), seen.end(), [](bool v) { return v; });
        });
        return std::vector<Frame>(received_.begin()+mark, received_.end());
    }
};

void standard_test(const Files& files) {
    FakeBuses buses({{{1,2,3,4}, {5,6,7,8}, {9,10,11,12}}});
    EncosDriver driver(files.mapping.string(), files.limits.string(), buses.paths());
    driver.connect();
    auto mapping = driver.calibration();
    auto state = driver.read(); auto start = now_ns();
    for (int id=1; id<=12; ++id) {
        int index = 12-id; double factor = (id % 2 ? 1.0 : -1.0)*1.5;
        check(mapping[index].raw_motor_id == id && near(mapping[index].sign * mapping[index].scale, factor), "mapping order or factor");
        check(near(mapping[index].model_min_rad, -30.0*1.5*pi/180) && near(mapping[index].model_max_rad, 30.0*1.5*pi/180), "model soft limits");
        check(state.received_ns[index] && state.received_ns[index] <= start && start-state.received_ns[index] <= 100000000, "feedback timestamp");
        check(near(state.raw_degrees[index], 0) && near(state.motor_rad[index], 0) && near(state.measured_current_a[index], id/10.0), "feedback axis or current");
    }
    buses.mit_feedback(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    state = driver.read();
    for (int id=1; id<=12; ++id) {
        auto idx = 12-id; double raw = (-5.0 + 32768.0*10.0/65535.0)*180.0/pi;
        check(near(state.raw_degrees[idx], raw) && near(state.measured_current_a[idx], -2.0+2048.0*4.0/4095.0), "MIT feedback decode");
    }
    buses.mit_feedback(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    throws([&]{ driver.enable_force(); }, "force before home must fail");
    auto home_mark = buses.mark(); driver.home();
    auto home_frames = buses.after(home_mark, 1);
    std::array<bool, 12> saw_home{};
    for (const auto& f : home_frames) {
        auto w = word(f); if ((w >> 61) != 1 || ((w >> 14) & 0x7fff) != 50) continue;
        check(((w >> 2) & 0xfff) == (f.id == 1 ? 15U : 10U), "homing current");
        saw_home[f.id-1] = true;
    }
    for (bool seen : saw_home) check(seen, "missing home command");
    Vec12 target{}; target.fill(0.25);
    auto mark = buses.mark(); auto before = now_ns(); auto deadline = before + 100000000;
    auto actual = driver.write_position(target, deadline); auto stamps = driver.last_command_write_ns();
    auto frames = buses.after(mark, 0); std::array<bool, 12> seen{};
    for (const auto& f : frames) {
        auto w = word(f); if ((w >> 61) != 0 || ((w >> 49) & 0xfff) == 0) continue;
        check(((w >> 49)&0xfff) == 369 && ((w >> 40)&0x1ff) == 51 && ((w >> 12)&0xfff) == 2048 && (w&0xfff) == 2048, "MIT position gains or feedforward encoding");
        int idx = 12-f.id; double factor = (f.id % 2 ? 1.0 : -1.0)*1.5;
        unsigned code = (w >> 24)&0xffff;
        check(near(actual[idx], (-5.0+code*10.0/65535.0)*factor), "position confirmation decoding");
        check(std::abs(actual[idx]-0.25) < 0.0003, "position sign or units");
        check(stamps[idx] >= before && stamps[idx] < deadline, "actual serial submission timestamp");
        seen[f.id-1] = true;
    }
    for (bool s : seen) check(s, "incomplete MIT position submissions");
    Vec12 beyond{}; beyond.fill(5.0);
    auto clipped = driver.write_position(beyond, now_ns()+100000000);
    for (double value : clipped)
        check(value <= 30.0*1.5*pi/180.0 && value > 0.78, "position soft-limit saturation");
    driver.write_position(target, now_ns()+100000000);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    driver.enable_force();
    Vec12 excessive{}; excessive.fill(5.0);
    auto limited = driver.write_torque(excessive, now_ns()+100000000);
    for (int id=1; id<=12; ++id) {
        double raw = limited[12-id] * (id % 2 ? 1.5 : -1.5);
        check(std::abs(raw) <= 1.0 && std::abs(raw) > 0.99 &&
              (id%2 ? raw > 0 : raw < 0), "software torque cap or sign");
    }
    // Reset physical feedback to soft boundaries and wait for actual serial feedback.
    driver.disable();
    for (int id=1; id<=12; ++id) buses.set_angle(id, id % 2 ? 30.0f : -30.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    state = driver.read();
    for (int id=1; id<=12; ++id) check(near(state.raw_degrees[12-id], id%2 ? 30.0 : -30.0), "boundary feedback");
    driver.enable_force();
    Vec12 torque{}; for (int id=1; id<=12; ++id) {
        double raw_outward = id % 2 ? 0.7 : -0.7;
        torque[12-id] = raw_outward / ((id % 2 ? 1.0 : -1.0)*1.5);
    }
    mark = buses.mark(); before = now_ns(); deadline = before + 100000000;
    auto confirmed = driver.write_torque(torque, deadline); stamps = driver.last_command_write_ns();
    frames = buses.after(mark, 3); seen.fill(false);
    for (const auto& f : frames) {
        auto w = word(f); if ((w >> 61) != 0 || ((w >> 49)&0xfff) != 0 || ((w >> 40)&0x1ff) != 0) continue;
        unsigned code = w&0xfff; double raw = -10.0+code*20.0/4095.0;
        double factor = (f.id%2 ? 1.0 : -1.0)*1.5;
        check((f.id%2 ? raw < 0 : raw > 0) && std::abs(raw) <= 1.0, "boundary torque projection or cap");
        check(near(confirmed[12-f.id], raw/factor), "quantized nominal torque confirmation");
        check(stamps[12-f.id] >= before && stamps[12-f.id] < deadline, "torque write timestamp");
        seen[f.id-1] = true;
    }
    for (bool s : seen) check(s, "incomplete MIT torque submissions");
    mark = buses.mark(); before = now_ns(); driver.disable();
    stamps = driver.last_command_write_ns();
    for (auto stamp : stamps)
        check(stamp >= before && stamp < before + 100000000, "stop must confirm each serial submission");
    frames = buses.after(mark, 2); seen.fill(false);
    for (const auto& f : frames) {
        auto w = word(f);
        if ((w>>61) == 1 && ((w>>14)&0x7fff) == 0 && ((w>>2)&0xfff) == 0) seen[f.id-1] = true;
    }
    for (bool s : seen) check(s, "disable did not deliver zero-current polling to all axes");
    buses.silence(7); std::this_thread::sleep_for(std::chrono::milliseconds(150));
    throws([&]{ driver.read(); }, "one stale axis must fail the whole frame");
    buses.silence(0); std::this_thread::sleep_for(std::chrono::milliseconds(80)); driver.read();
    buses.fault(4); std::this_thread::sleep_for(std::chrono::milliseconds(60));
    throws([&]{ driver.read(); }, "motor fault feedback must fail");
    buses.fault(0); std::this_thread::sleep_for(std::chrono::milliseconds(60)); driver.read();
    throws([&]{ driver.write_position(target, now_ns()-1); }, "expired command must fail");
    throws([&]{ driver.write_position(target, now_ns()+100000000); }, "expired command must latch output fault");
    driver.close();
}

void partial_submission_test(const Files& files) {
    // Ten motors on one bus require at least 45ms to reach all axes at 5ms per slot.
    FakeBuses buses({{{1,2,3,4,5,6,7,8,9,10}, {11}, {12}}});
    EncosDriver driver(files.mapping.string(), files.limits.string(), buses.paths());
    driver.connect(); driver.home();
    Vec12 target{}; target.fill(0.1);
    auto mark = buses.mark();
    auto begin = now_ns(); auto deadline = begin + 30000000;
    throws([&]{ driver.write_position(target, deadline); }, "partial frame must fail by deadline");
    auto writes = driver.last_command_write_ns();
    int count = 0;
    for (auto stamp : writes) if (stamp >= begin) { check(stamp < deadline, "write after deadline"); ++count; }
    check(count > 0 && count < 12, "expected a partially submitted frame");
    throws([&]{ driver.write_position(target, now_ns()+100000000); }, "partial frame must latch output fault");
    mark = buses.mark();
    driver.close();
    auto frames = buses.after(mark, 2, 150); std::array<bool, 12> stopped{};
    for (const auto& f : frames) {
        auto w=word(f); if ((w>>61)==1 && ((w>>14)&0x7fff)==0 && ((w>>2)&0xfff)==0) stopped[f.id-1]=true;
    }
    for (bool s : stopped) check(s, "close did not submit passive command for each axis");
}

void duplicate_scan_test(const Files& files) {
    FakeBuses buses({{{1,2,3,4}, {5,6,7,8}, {8,9,10,11}}});
    EncosDriver driver(files.mapping.string(), files.limits.string(), buses.paths());
    throws([&] { driver.connect(); }, "duplicate scanned motor ID must fail");
}

void stop_failure_test(const Files& files) {
    FakeBuses buses({{{1,2,3,4}, {5,6,7,8}, {9,10,11,12}}});
    EncosDriver driver(files.mapping.string(), files.limits.string(), buses.paths());
    driver.connect(); driver.home();
    Vec12 command{}; command.fill(0.1);
    driver.write_position(command, now_ns()+100000000);
    buses.disconnect(1);
    throws([&] { driver.disable(); }, "lost channel must fail stop confirmation");
    throws([&] { driver.write_position(command, now_ns()+100000000); }, "failed stop must latch");
    bool stop_error = false, close_error = false;
    try { driver.close(); }
    catch (const std::exception& e) {
        std::string message = e.what();
        stop_error = message.find("stop") != std::string::npos;
        close_error = message.find("close command failed") != std::string::npos;
    }
    check(stop_error && close_error, "explicit close must report stop and SLCAN C failures");
}
int main() {
    try { Files files; duplicate_scan_test(files); standard_test(files); partial_submission_test(files); stop_failure_test(files); std::cout << "Encos offline protocol tests passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << "Encos offline protocol test failed: " << e.what() << '\n'; return 1; }
}
