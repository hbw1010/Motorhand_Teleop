// motorhand_glove_native: one Motorhand exoskeleton glove -> Wuji Hand 2 targets, in C++.
//
// The same worker as motorhand_glove/worker.py, with the same command line and the same JSON lines on stdout,
// built from this repository's own C++ (native_control): the CAN driver (here with --slot-ms), the vE3.3 model
// and the WujiRetarget, as its motorhand_native teleop runs them. No Python, so no GIL between the CAN threads
// and the solve, and a frame takes ~0.5 ms instead of 6-8. Without --follow or --feedback-limits the glove is only
// read: no motor enabled or driven. With them it is homed first, then driven as glove_control.hpp says, from the
// Wuji hand's state and final target the node writes on stdin (HandFeed).
//
//   {"ready": {...}}             connected: three adapters, all 12 motors
//   {"ns": <ns>, "q": [20]}      a target, from the glove readings whose oldest is at monotonic <ns>
//   {"status": "<why>"}          not streaming; it reconnects every --retry s and says so again only on a new why
//   {"stats": {...}}             every --stats s: loop_hz, age_ms (the oldest reading's age when its target goes
//                                out), span_ms (oldest to newest of a frame's 12), solve_ms (model + retarget),
//                                motor_hz (how often each motor's reading changed, as this loop saw it), pinch_cm
//                                (the closest index, middle, ring and pinky tips came to the thumb's, as the
//                                retarget saw them); glove: the output's (glove_control.cpp)
//   {"event": "<what>"}          the glove's output: homing, its mode changing
// --pinch-d1/--pinch-d2 (cm) and --pinch-weight set the WujiRetarget's pinch: below d2 a finger's tip is pulled to the
// thumb's with weight (d2 - distance) / (d2 - d1), capped at the weight (Motorhand's own: its yaml's 2 and 4 cm, 0.7;
// 0 or not given: the yaml's). Every WujiRetarget here (this loop's, the Jacobian's, the inverse's) loads one yaml:
// Motorhand's own for the side, or with --pinch-* a copy of it with those settings (derived_yaml below).
// --follow: while the hand follows something else, the glove follows the hand, with --follow-kp its stiffness (MIT kp,
// Nm/rad; Motorhand's 45 when not given), its push on each motor within that motor's force-feedback cap (the
// --feedback-limits'), or --follow-max-nm on every motor without them (2 when not given). --feedback-limits <json>: while it
// follows this glove, force feedback, with --feedback-contact <json> (empty: Motorhand's own for the side) and
// --wuji-serial (the hand's, checked against the limits'). The glove is homed (Motorhand's driver drives nothing
// before) once per worker: a reconnect resumes from where the hand is; --no-home: not at all.
#include "glove_control.hpp"
#include "encos_driver.hpp"
#include "ve33_model.hpp"
#include "wuji_retarget.hpp"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace motorhand::native;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

std::atomic<bool> stopping{false};
extern "C" void on_sigterm(int) { stopping.store(true); }  // a flag only: a lock here can deadlock the loop

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

struct Args {
    std::string side = "right";
    fs::path root, mapping;
    double rate = 100.0, slot_ms = 5.0, stats_s = 10.0, retry_s = 1.0;
    double pinch_d1 = 0.0, pinch_d2 = 0.0, pinch_weight = 0.0;  // 0: the retarget yaml's
    double follow_kp = 45.0, follow_max_nm = 2.0;
    bool follow = false, home = true;
    fs::path feedback_limits, feedback_contact;
    std::string wuji_serial;
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
    std::array<std::string, 3> test_ports;  // the simulated adapters (PTYs); tests only
#endif
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument(flag + " needs a value");
            return argv[i];
        };
        if (flag == "--side") a.side = value();
        else if (flag == "--root") a.root = value();
        else if (flag == "--rate") a.rate = std::stod(value());
        else if (flag == "--slot-ms") a.slot_ms = std::stod(value());
        else if (flag == "--stats") a.stats_s = std::stod(value());
        else if (flag == "--retry") a.retry_s = std::stod(value());
        else if (flag == "--mapping") a.mapping = value();
        else if (flag == "--pinch-d1") a.pinch_d1 = std::stod(value());
        else if (flag == "--pinch-d2") a.pinch_d2 = std::stod(value());
        else if (flag == "--pinch-weight") a.pinch_weight = std::stod(value());
        else if (flag == "--follow") a.follow = true;
        else if (flag == "--follow-kp") a.follow_kp = std::stod(value());
        else if (flag == "--follow-max-nm") a.follow_max_nm = std::stod(value());
        else if (flag == "--no-home") a.home = false;
        else if (flag == "--feedback-limits") a.feedback_limits = value();
        else if (flag == "--feedback-contact") a.feedback_contact = value();
        else if (flag == "--wuji-serial") a.wuji_serial = value();
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
        else if (flag == "--test-ports") {
            std::string list = value();
            for (auto& port : a.test_ports) {
                const auto comma = list.find(',');
                port = list.substr(0, comma);
                list = comma == std::string::npos ? "" : list.substr(comma + 1);
            }
        }
#endif
        else if (flag == "--ports") {
            if (!value().empty())
                throw std::invalid_argument("--ports: the C++ CAN driver finds its three CANable2 itself");
        } else throw std::invalid_argument("unknown option " + flag);
    }
    if (a.side != "left" && a.side != "right") throw std::invalid_argument("--side left|right");
    if (a.root.empty()) throw std::invalid_argument("--root: the Motorhand_FFB checkout");
    if (!(a.rate > 0) || !(a.retry_s > 0) || a.stats_s < 0)
        throw std::invalid_argument("--rate and --retry must be positive, --stats not negative");
    // Each motor (4 an adapter) is read every 4 slots and its reading lasts the driver's 100 ms: over 25 ms none would.
    if (!(a.slot_ms >= 0.001 && a.slot_ms <= 25))
        throw std::invalid_argument("--slot-ms in [0.001, 25] ms");
    if (a.pinch_d1 < 0 || a.pinch_d2 < 0 || (a.pinch_d1 > 0 && a.pinch_d2 > 0 && a.pinch_d1 >= a.pinch_d2) ||
        !(a.pinch_weight >= 0 && a.pinch_weight <= 1))
        throw std::invalid_argument("--pinch-d1 < --pinch-d2 (cm, 0: the yaml's), --pinch-weight in (0, 1] (0: the yaml's)");
    if (!(a.follow_kp > 0 && a.follow_kp <= 45))
        throw std::invalid_argument("--follow-kp in (0, 45] Nm/rad");
    if (!(a.follow_max_nm > 0 && a.follow_max_nm <= 10))
        throw std::invalid_argument("--follow-max-nm in (0, 10] Nm");
    if (a.mapping.empty()) a.mapping = a.root / "hardware" / "encos_dexhand" / "configs" / "glove_mapping.json";
    if (!a.feedback_limits.empty() && a.feedback_contact.empty())
        a.feedback_contact = a.root / "config" / "motorhand_feedback" / ("contact-impedance-" + a.side + ".json");
    return a;
}

// Motorhand's own retarget yaml for the side.
fs::path official_yaml(const Args& a) {
    return fs::absolute(a.root / "config" / "retargeting" / "official" / (a.side + ".yaml"));
}

// With --pinch-*: a copy of Motorhand's own yaml with those pinch settings, its urdf_path made absolute (it is
// relative to the yaml), in a file of this worker's own in the temp dir: created new (mkstemp: no other worker's
// settings can replace it, nor a link redirect it), written whole before anything reads it, removed when the worker
// ends (RemoveOnExit). The retargets of a GloveControl rebuilt after a motor fault read it again. Empty: no copy.
fs::path derived_yaml(const Args& a) {
    if (!(a.pinch_d1 > 0) && !(a.pinch_d2 > 0) && !(a.pinch_weight > 0)) return {};
    const fs::path official = official_yaml(a);
    YAML::Node root = YAML::LoadFile(official.string());
    YAML::Node optimizer = root["optimizer"];
    const fs::path urdf = optimizer["urdf_path"].as<std::string>();
    if (urdf.is_relative()) optimizer["urdf_path"] = (official.parent_path() / urdf).lexically_normal().string();
    YAML::Node retarget = root["retarget"];
    for (const char* finger : {"index", "middle", "ring", "pinky"}) {
        if (a.pinch_d1 > 0) retarget["pinch_thresholds"][finger]["d1"] = a.pinch_d1;
        if (a.pinch_d2 > 0) retarget["pinch_thresholds"][finger]["d2"] = a.pinch_d2;
    }
    if (a.pinch_weight > 0) retarget["pinch_weight_max"] = a.pinch_weight;
    YAML::Emitter emitter;
    emitter << root;
    const std::string text = std::string(emitter.c_str()) + "\n";
    std::string path = (fs::temp_directory_path() / ("motorhand_glove_retarget_" + a.side + "_XXXXXX")).string();
    const int fd = ::mkstemp(path.data());
    if (fd < 0) throw std::runtime_error("cannot create a retarget yaml in " + fs::temp_directory_path().string());
    bool written = true;
    for (std::size_t done = 0; written && done < text.size();) {
        const ssize_t n = ::write(fd, text.data() + done, text.size() - done);
        if (n < 0 && errno == EINTR) continue;
        written = n > 0;
        if (written) done += static_cast<std::size_t>(n);
    }
    written = ::close(fd) == 0 && written;
    if (!written) {
        std::error_code ignored;
        fs::remove(path, ignored);
        throw std::runtime_error("cannot write the retarget yaml " + path);
    }
    return path;
}

struct RemoveOnExit {
    fs::path path;
    ~RemoveOnExit() {
        std::error_code ignored;
        if (!path.empty()) fs::remove(path, ignored);
    }
};

// The JSON lines go to the original stdout; anything else (the libraries' prints) to stderr.
class Protocol {
public:
    Protocol() {
        const int fd = ::dup(STDOUT_FILENO);
        ::dup2(STDERR_FILENO, STDOUT_FILENO);
        out_ = ::fdopen(fd, "w");
        if (!out_) throw std::runtime_error("cannot open the protocol stream");
    }
    void emit(const nlohmann::json& message) {
        const std::string line = message.dump() + "\n";
        std::fwrite(line.data(), 1, line.size(), out_);
        std::fflush(out_);
    }

private:
    std::FILE* out_ = nullptr;
};

void pause(double seconds) {
    const auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    while (!stopping.load() && Clock::now() < end)
        std::this_thread::sleep_for(std::min<Clock::duration>(std::chrono::milliseconds(50), end - Clock::now()));
}

double median(std::vector<double> values) {
    if (values.empty()) return std::nan("");
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

double rounded(double value) { return std::round(value * 10.0) / 10.0; }

class Stats {
public:
    explicit Stats(double every_s) : every_s_(every_s), since_(Clock::now()) {}
    void add(const GloveFrame& frame, const Points& points, std::uint64_t sent_ns, double solve_ms) {
        ++frames_;
        for (int finger = 1; finger < 5; ++finger) {  // the canonical 21 points: 4 the thumb's tip, 8/12/16/20 the others'
            double squared = 0;
            for (int k = 0; k < 3; ++k) {
                const double d = points[3 * (4 + 4 * finger) + k] - points[3 * 4 + k];
                squared += d * d;
            }
            closest_[finger - 1] = std::min(closest_[finger - 1], 100.0 * std::sqrt(squared));
        }
        const auto [lo, hi] = std::minmax_element(frame.received_ns.begin(), frame.received_ns.end());
        ages_.push_back((sent_ns - frame.oldest_ns) / 1e6);
        spans_.push_back((*hi - *lo) / 1e6);
        solves_.push_back(solve_ms);
        for (int i = 0; i < 12; ++i) {
            if (frame.received_ns[i] != last_[i]) ++changes_[i];
            last_[i] = frame.received_ns[i];
        }
    }
    std::optional<nlohmann::json> due() {
        const double seconds = std::chrono::duration<double>(Clock::now() - since_).count();
        if (every_s_ <= 0 || seconds < every_s_) return std::nullopt;
        std::vector<double> motor_hz;
        for (int i = 0; i < 12; ++i) motor_hz.push_back(changes_[i] / seconds);
        nlohmann::json stats = {
            {"loop_hz", rounded(frames_ / seconds)}, {"age_ms", rounded(median(ages_))},
            {"span_ms", rounded(median(spans_))}, {"solve_ms", std::round(median(solves_) * 100.0) / 100.0},
            {"motor_hz", rounded(median(motor_hz))},
            {"slowest_motor_hz", rounded(*std::min_element(motor_hz.begin(), motor_hz.end()))}};
        std::vector<double> pinch;
        for (double cm : closest_) pinch.push_back(std::isfinite(cm) ? rounded(cm) : -1.0);
        stats["pinch_cm"] = pinch;
        const auto seen = last_;  // a reading counts once, across windows too
        *this = Stats(every_s_);
        last_ = seen;
        return stats;
    }

private:
    double every_s_;
    Clock::time_point since_;
    long frames_ = 0;
    std::vector<double> ages_, spans_, solves_;
    std::array<long, 12> changes_{};
    std::array<std::uint64_t, 12> last_{};
    std::array<double, 4> closest_{inf(), inf(), inf(), inf()};  // cm, the closest each tip came to the thumb's
    static constexpr double inf() { return std::numeric_limits<double>::infinity(); }
};
}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "motorhand_glove_native: %s\n", e.what());
        return 2;
    }
    Protocol protocol;
    ::prctl(PR_SET_PDEATHSIG, SIGTERM);  // the node's process gone: let the glove's adapters go too
    if (::getppid() == 1) return 0;
    std::signal(SIGTERM, on_sigterm);
    std::signal(SIGINT, SIG_IGN);  // the node stops it; a Ctrl+C in the node's terminal is the node's
    motorhand_glove::set_skip_homing(!args.home);

    const bool left = args.side == "left";
    const bool output = args.follow || !args.feedback_limits.empty();
    motorhand_glove::HandFeed feed;
    if (output) {  // the node's lines; it never ends while the node runs, so not joined
        std::thread([&feed] {
            // read(), not std::cin: blocked in it, stdio's stdin lock would stall exit()'s flush of all streams
            std::string pending;
            char buffer[65536];
            for (ssize_t n; (n = ::read(STDIN_FILENO, buffer, sizeof buffer)) != 0;) {
                if (n < 0) {
                    if (errno == EINTR) continue;
                    return;
                }
                pending.append(buffer, static_cast<std::size_t>(n));
                for (std::size_t end; (end = pending.find('\n')) != std::string::npos; pending.erase(0, end + 1))
                    feed.take(pending.substr(0, end));
            }
        }).detach();
    }
    motorhand_glove::ControlOptions control_options;
    control_options.root = args.root;
    control_options.left = left;
    control_options.follow = args.follow;
    control_options.follow_max_nm = args.follow_max_nm;
    control_options.follow_kp = args.follow_kp;
    control_options.limits = args.feedback_limits;
    control_options.contact = args.feedback_contact;
    control_options.serial = args.wuji_serial;
    RemoveOnExit derived;  // before the retargets: they go first
    std::optional<Ve33Model> model;
    std::optional<WujiRetarget> retarget;
    try {
        model.emplace(args.root / "assets" / "hand_motor_ve33", left);
        derived.path = derived_yaml(args);
        control_options.retarget_yaml = derived.path.empty() ? official_yaml(args) : derived.path;
        retarget.emplace(control_options.retarget_yaml, left);
    } catch (const std::exception& e) {
        protocol.emit({{"status", std::string("the glove model or retarget did not load: ") + e.what()}});
        return 1;
    }
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / args.rate));
    std::string said;
    while (!stopping.load()) {
        try {
#ifdef MOTORHAND_ENCOS_OFFLINE_TEST
            EncosDriver glove(args.mapping.string(), args.feedback_limits.string(),
                              EncosDriver::OfflineTestPorts{args.test_ports});
#else
            EncosDriver glove(args.mapping.string(), args.feedback_limits.string());
#endif
            glove.set_slot(std::chrono::microseconds(std::llround(args.slot_ms * 1000.0)));
            glove.set_position_kp(args.follow_kp);
            glove.connect();
            std::vector<int> online(12);
            for (int i = 0; i < 12; ++i) online[i] = i + 1;
            protocol.emit({{"ready", {{"online_ids", online}, {"mapping", args.mapping.string()},
                                      {"slot_ms", args.slot_ms}, {"solver", "native"},
                                      {"follow", args.follow}, {"follow_kp", args.follow_kp}, {"force_feedback", !args.feedback_limits.empty()}}}});
            std::optional<motorhand_glove::GloveControl> control;  // after glove: stopped before it closes
            if (output) control.emplace(glove, control_options);
            said.clear();
            Stats stats(args.stats_s);
            std::optional<GlovePose> last;
            auto due = Clock::now();
            while (!stopping.load()) {
                const GloveFrame frame = glove.read();
                const auto started = Clock::now();
                GlovePose pose = model->evaluate_motors(frame.motor_rad, last ? &*last : nullptr);
                Vec20 q = retarget->solve(pose.points_m);
                for (int spread : {5, 9, 13, 17}) q[spread] = 0.0;  // the glove has no spread input
                const double solve_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                protocol.emit({{"ns", frame.oldest_ns}, {"q", q}});
                stats.add(frame, pose.points_m, now_ns(), solve_ms);
                if (control) {
                    control->step(frame, pose, feed.snapshot());
                    for (const auto& event : control->events()) protocol.emit({{"event", event}});
                }
                last = std::move(pose);
                if (auto report = stats.due()) {
                    if (control) (*report)["glove"] = control->stats();
                    protocol.emit({{"stats", *report}});
                }
                due += period;
                const auto now = Clock::now();
                if (due > now) std::this_thread::sleep_until(due);
                else due = now;  // behind: carry on from now, no burst
            }
            control.reset();
            glove.close();
        } catch (const std::exception& e) {
            const std::string reason = e.what();
            if (reason != said) protocol.emit({{"status", reason}});
            said = reason;
        }
        if (!stopping.load()) pause(args.retry_s);
    }
    return 0;
}
