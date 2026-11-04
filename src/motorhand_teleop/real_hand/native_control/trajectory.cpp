#include "trajectory.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <sys/file.h>

namespace motorhand::native {
namespace {
using Json = nlohmann::ordered_json;
constexpr std::array<const char*, 20> joints = {
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_mcp_abd", "index_pip", "index_dip",
    "middle_mcp_flex", "middle_mcp_abd", "middle_pip", "middle_dip",
    "ring_mcp_flex", "ring_mcp_abd", "ring_pip", "ring_dip",
    "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip"};

void identity(const std::string& hand, const std::string& serial, const std::string& firmware) {
    auto valid = [](const std::string& s) {
        return s.find('\0') == std::string::npos &&
            s.find_first_not_of(" \t\n\r\f\v") != std::string::npos;
    };
    if ((hand != "left" && hand != "right") || !valid(serial) || !valid(firmware))
        throw std::runtime_error("invalid trajectory identity");
}
void finite(const Vec20& v) {
    for (double x : v) if (!std::isfinite(x)) throw std::runtime_error("non-finite trajectory vector");
}
Vec20 vector(const Json& value) {
    if (!value.is_array() || value.size() != 20) throw std::runtime_error("invalid trajectory vector");
    Vec20 result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!value[i].is_number()) throw std::runtime_error("invalid trajectory vector element");
        result[i] = value[i].get<double>();
    }
    finite(result);
    return result;
}
std::uint64_t integer(const Json& v) {
    if (!v.is_number_unsigned() && !v.is_number_integer()) throw std::runtime_error("invalid trajectory integer");
    if (v.is_number_integer() && !v.is_number_unsigned() && v.get<std::int64_t>() < 0)
        throw std::runtime_error("negative trajectory integer");
    return v.get<std::uint64_t>();
}
class Digest {
    EVP_MD_CTX* ctx_ = EVP_MD_CTX_new();
public:
    Digest() { if (!ctx_ || EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) != 1) { EVP_MD_CTX_free(ctx_); throw std::runtime_error("SHA256 initialization failed"); } }
    ~Digest() { EVP_MD_CTX_free(ctx_); }
    void add(const std::string& s) { if (EVP_DigestUpdate(ctx_, s.data(), s.size()) != 1) throw std::runtime_error("SHA256 update failed"); }
    std::string finish() {
        unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned int n = 0;
        if (EVP_DigestFinal_ex(ctx_, bytes, &n) != 1 || n != 32) throw std::runtime_error("SHA256 finalization failed");
        static constexpr char hex[] = "0123456789abcdef";
        std::string out; out.reserve(64);
        for (unsigned i = 0; i < n; ++i) { out.push_back(hex[bytes[i] >> 4]); out.push_back(hex[bytes[i] & 15]); }
        return out;
    }
};
Json parse(const std::string& line) {
    std::vector<std::unordered_set<std::string>> keys;
    auto callback = [&keys](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        else if (event == Json::parse_event_t::key) {
            if (keys.empty() || !keys.back().insert(parsed.get<std::string>()).second)
                throw std::runtime_error("duplicate trajectory JSON key");
        } else if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    };
    return Json::parse(line, callback);
}
std::string line(const Json& obj) { return obj.dump(-1, ' ', false, Json::error_handler_t::strict) + "\n"; }
void write_all(int fd, const std::string& data) {
    std::size_t done = 0;
    while (done < data.size()) {
        const auto n = ::write(fd, data.data() + done, data.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("trajectory write failed: " + std::string(std::strerror(errno)));
        done += static_cast<std::size_t>(n);
    }
}
Json header(const std::string& hand, const std::string& serial, const std::string& firmware) {
    return Json{{"type", "header"}, {"version", 1}, {"hand", hand}, {"serial_number", serial},
                {"firmware_version", firmware}, {"joint_names", joints}, {"time_unit", "ns"},
                {"position_unit", "rad"}, {"velocity_unit", "rad/s"}, {"effort_unit", "A"}};
}
} // namespace

Trajectory::Trajectory(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open trajectory");
    Digest digest;
    std::string text;
    bool have_header = false, have_end = false;
    std::uint64_t first = 0, previous = 0;
    while (std::getline(input, text)) {
        if (input.eof()) throw std::runtime_error("truncated trajectory line");
        text += '\n';
        const auto obj = parse(text);
        if (!obj.is_object() || have_end) throw std::runtime_error("invalid trajectory record order");
        const auto kind = obj.at("type").get<std::string>();
        if (!have_header) {
            if (kind != "header" || integer(obj.at("version")) != 1) throw std::runtime_error("invalid trajectory header");
            hand_ = obj.at("hand").get<std::string>(); serial_ = obj.at("serial_number").get<std::string>();
            firmware_ = obj.at("firmware_version").get<std::string>(); identity(hand_, serial_, firmware_);
            if (obj.at("joint_names") != Json(joints) || obj.at("time_unit") != "ns" ||
                obj.at("position_unit") != "rad" || obj.at("velocity_unit") != "rad/s" || obj.at("effort_unit") != "A")
                throw std::runtime_error("trajectory joint order or units mismatch");
            have_header = true;
        } else if (kind == "frame") {
            const auto stamp = integer(obj.at("source_ns")), elapsed = integer(obj.at("elapsed_ns"));
            if (!stamp || (previous && stamp <= previous)) throw std::runtime_error("invalid trajectory timestamp order");
            if (!first) first = stamp;
            if (elapsed != stamp - first) throw std::runtime_error("inconsistent trajectory elapsed time");
            previous = stamp;
            positions_.push_back(vector(obj.at("positions_rad")));
            for (const char* key : {"velocities_rad_s", "effort_a"}) if (!obj.at(key).is_null()) vector(obj.at(key));
            vector(obj.at("target_positions_rad"));
            times_.push_back(elapsed);
        } else if (kind == "end") {
            if (times_.size() < 2 || !times_.back() || integer(obj.at("frame_count")) != times_.size() ||
                integer(obj.at("duration_ns")) != times_.back() || obj.at("sha256") != digest.finish())
                throw std::runtime_error("invalid trajectory completion or checksum");
            have_end = true;
            continue;
        } else throw std::runtime_error("invalid trajectory record");
        digest.add(text);
    }
    if (input.bad() || !have_header || !have_end) throw std::runtime_error("incomplete trajectory");
}
std::uint64_t Trajectory::duration_ns() const { return times_.back(); }
void Trajectory::validate_limits(const Vec20& lower, const Vec20& upper) const {
    for (std::size_t axis = 0; axis < lower.size(); ++axis) {
        if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || lower[axis] > upper[axis])
            throw std::runtime_error("invalid device limits for axis " + std::to_string(axis));
    }
    for (std::size_t frame = 0; frame < positions_.size(); ++frame) {
        for (std::size_t axis = 0; axis < lower.size(); ++axis) {
            const double value = positions_[frame][axis];
            if (!std::isfinite(value) || value < lower[axis] || value > upper[axis])
                throw std::runtime_error("trajectory frame " + std::to_string(frame) +
                                         " axis " + std::to_string(axis) + " (" + joints[axis] +
                                         ") outside device limits");
        }
    }
}
Vec20 Trajectory::at(std::uint64_t elapsed) const {
    if (elapsed >= times_.back()) return positions_.back();
    auto upper = std::upper_bound(times_.begin(), times_.end(), elapsed);
    const auto i = static_cast<std::size_t>(upper - times_.begin());
    const double w = static_cast<double>(elapsed - times_[i - 1]) / static_cast<double>(times_[i] - times_[i - 1]);
    Vec20 result{};
    for (std::size_t j = 0; j < result.size(); ++j) result[j] = positions_[i - 1][j] * (1.0 - w) + positions_[i][j] * w;
    return result;
}
std::string Trajectory::hand() const { return hand_; }
std::string Trajectory::serial() const { return serial_; }
std::string Trajectory::firmware() const { return firmware_; }

struct TrajectoryRecorder::Impl {
    static constexpr std::size_t capacity = 1024;
    struct Item { TrajectoryFrame frame; std::uint64_t elapsed; };
    std::array<Item, capacity> queue{};
    std::atomic<std::size_t> head{0}, tail{0};
    std::atomic<bool> stopping{false}, failed{false}, writer_failed{false};
    std::mutex mutex;
    std::condition_variable wake;
    std::thread writer;
    int fd = -1;
    bool closed = false;
    std::uint64_t first = 0, last = 0, count = 0;
    std::string header_line;

    Impl(const std::filesystem::path& path, const std::string& hand, const std::string& serial, const std::string& firmware) {
        identity(hand, serial, firmware);
        header_line = line(header(hand, serial, firmware));
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (fd < 0) throw std::runtime_error("cannot open trajectory recording");
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); fd = -1; throw std::runtime_error("trajectory recording is locked"); }
        if (::ftruncate(fd, 0) != 0) { ::close(fd); fd = -1; throw std::runtime_error("cannot truncate trajectory recording"); }
        try { writer = std::thread([this] { run(); }); } catch (...) { ::close(fd); fd = -1; throw; }
    }
    ~Impl() { if (!closed) { try { stop(false); } catch (...) {} } }
    void run() noexcept {
        try {
            Digest digest;
            write_all(fd, header_line); digest.add(header_line);
            for (;;) {
                const auto t = tail.load(std::memory_order_relaxed);
                if (t != head.load(std::memory_order_acquire)) {
                    const Item item = queue[t % capacity];
                    tail.store(t + 1, std::memory_order_release);
                    const auto& f = item.frame;
                    Json obj{{"type", "frame"}, {"source_ns", f.source_ns}, {"elapsed_ns", item.elapsed},
                             {"positions_rad", f.position_rad}, {"velocities_rad_s", f.has_velocity ? Json(f.velocity_rad_s) : Json(nullptr)},
                             {"effort_a", f.has_effort ? Json(f.effort_a) : Json(nullptr)}, {"target_positions_rad", f.target_rad}};
                    auto bytes = line(obj); write_all(fd, bytes); digest.add(bytes);
                } else if (stopping.load(std::memory_order_acquire)) break;
                else { std::unique_lock<std::mutex> lock(mutex); wake.wait_for(lock, std::chrono::milliseconds(10)); }
            }
            if (!failed.load(std::memory_order_acquire) && count >= 2) {
                write_all(fd, line(Json{{"type", "end"}, {"frame_count", count}, {"duration_ns", last - first}, {"sha256", digest.finish()}}));
            }
        } catch (...) { writer_failed.store(true, std::memory_order_release); failed.store(true, std::memory_order_release); }
        if (::close(fd) != 0) writer_failed.store(true, std::memory_order_release);
        fd = -1;
    }
    void record(const TrajectoryFrame& f) {
        if (closed || failed.load(std::memory_order_acquire) || writer_failed.load(std::memory_order_acquire))
            throw std::runtime_error("trajectory recorder closed or failed");
        try {
            if (!f.source_ns) throw std::runtime_error("invalid source timestamp");
            if (count && f.source_ns == last) return;
            if (count && f.source_ns < last) throw std::runtime_error("source timestamp went backwards");
            finite(f.position_rad); finite(f.target_rad);
            if (f.has_velocity) finite(f.velocity_rad_s);
            if (f.has_effort) finite(f.effort_a);
            const auto h = head.load(std::memory_order_relaxed);
            if (h - tail.load(std::memory_order_acquire) >= capacity) throw std::runtime_error("trajectory recorder queue overflow");
            const auto base = count ? first : f.source_ns;
            queue[h % capacity] = Item{f, f.source_ns - base};
            first = base; last = f.source_ns; ++count;
            head.store(h + 1, std::memory_order_release);
            wake.notify_one();
        } catch (...) { failed.store(true, std::memory_order_release); throw; }
    }
    void stop(bool commit) {
        if (closed) return;
        closed = true;
        if (!commit || count < 2) failed.store(true, std::memory_order_release);
        stopping.store(true, std::memory_order_release); wake.notify_one();
        writer.join();
        if (writer_failed.load(std::memory_order_acquire)) throw std::runtime_error("trajectory writer failed");
        if (commit && failed.load(std::memory_order_acquire)) throw std::runtime_error("trajectory recording failed or has fewer than two distinct frames");
    }
};
TrajectoryRecorder::TrajectoryRecorder(const std::filesystem::path& path, std::string hand, std::string serial, std::string firmware)
    : impl_(std::make_unique<Impl>(path, hand, serial, firmware)) {}
TrajectoryRecorder::~TrajectoryRecorder() = default;
void TrajectoryRecorder::record(const TrajectoryFrame& frame) { impl_->record(frame); }
void TrajectoryRecorder::finish() { impl_->stop(true); }
void TrajectoryRecorder::abort() { impl_->stop(false); }
} // namespace motorhand::native
