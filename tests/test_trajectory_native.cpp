#include "../src/motorhand_teleop/real_hand/native_control/trajectory.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <string>

using motorhand::native::Trajectory;
using motorhand::native::TrajectoryFrame;
using motorhand::native::TrajectoryRecorder;
using motorhand::native::Vec20;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F&& f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
struct TempDir {
    fs::path path;
    TempDir() {
        std::string pattern = (fs::temp_directory_path() / "motorhand-trajectory-XXXXXX").string();
        char* result = ::mkdtemp(pattern.data());
        if (!result) throw std::runtime_error("mkdtemp failed");
        path = result;
    }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};
std::string read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("open failed");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
void put_bytes(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("write failed");
}
std::string sha256(const std::string& data) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned size = 0;
    if (EVP_Digest(data.data(), data.size(), digest, &size, EVP_sha256(), nullptr) != 1 || size != 32)
        throw std::runtime_error("SHA256 failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned i = 0; i < size; ++i) {
        result.push_back(hex[digest[i] >> 4]);
        result.push_back(hex[digest[i] & 15]);
    }
    return result;
}
TrajectoryFrame frame(std::uint64_t stamp, double base, bool optional) {
    TrajectoryFrame result{};
    result.source_ns = stamp;
    result.has_velocity = result.has_effort = optional;
    for (std::size_t i = 0; i < 20; ++i) {
        result.position_rad[i] = base + static_cast<double>(i) / 100.0;
        result.target_rad[i] = result.position_rad[i] + 0.2;
        result.velocity_rad_s[i] = static_cast<double>(i) / 10.0;
        result.effort_a[i] = -static_cast<double>(i) / 20.0;
    }
    return result;
}
void near(double actual, double expected) {
    require(std::abs(actual - expected) < 1e-12, "incorrect interpolated position");
}
void test(const fs::path& dir) {
    const auto path = dir / "trajectory.jsonl";
    const auto first = frame(1'000, 0.0, true);
    const auto second = frame(2'000, 1.0, false);
    const auto third = frame(3'000, 0.0, true);
    {
        TrajectoryRecorder recorder(path, "right", "serial", "firmware");
        recorder.record(first);
        auto duplicate = frame(1'000, 9.0, false);
        recorder.record(duplicate);
        recorder.record(second);
        recorder.record(third);
        recorder.finish();
    }
    const auto bytes = read_bytes(path);
    require(bytes.find("\"frame_count\":3") != std::string::npos, "duplicate was recorded");
    require(bytes.find("\"velocities_rad_s\":null") != std::string::npos &&
            bytes.find("\"effort_a\":null") != std::string::npos, "optional values not encoded");
    Trajectory trajectory(path);
    require(trajectory.hand() == "right" && trajectory.serial() == "serial" && trajectory.firmware() == "firmware", "identity mismatch");
    require(trajectory.duration_ns() == 2'000, "incorrect duration");
    for (std::size_t i = 0; i < 20; ++i) {
        near(trajectory.at(0)[i], first.position_rad[i]);
        near(trajectory.at(500)[i], (first.position_rad[i] + second.position_rad[i]) / 2);
        near(trajectory.at(1'000)[i], second.position_rad[i]);
        near(trajectory.at(1'500)[i], (second.position_rad[i] + third.position_rad[i]) / 2);
        near(trajectory.at(9'000)[i], third.position_rad[i]);
    }
    Vec20 lower{}, upper{};
    lower.fill(-1.0); upper.fill(2.0);
    trajectory.validate_limits(lower, upper);
    upper[7] = 0.9;
    bool identified = false;
    try { trajectory.validate_limits(lower, upper); }
    catch (const std::exception& e) {
        const std::string message = e.what();
        identified = message.find("frame 1") != std::string::npos && message.find("axis 7") != std::string::npos;
    }
    require(identified, "interior frame limit violation not identified");

    auto corrupt = bytes;
    const auto serial = corrupt.find("\"serial\"");
    require(serial != std::string::npos, "missing serial in recording");
    corrupt[serial + 1] = 'S';
    put_bytes(dir / "corrupt.jsonl", corrupt);
    rejects([&] { Trajectory t(dir / "corrupt.jsonl"); }, "corrupt checksum accepted");
    put_bytes(dir / "truncated.jsonl", bytes.substr(0, bytes.size() - 1));
    rejects([&] { Trajectory t(dir / "truncated.jsonl"); }, "truncated footer accepted");
    const auto footer = bytes.rfind("{\"type\":\"end\"");
    require(footer != std::string::npos, "missing footer");
    put_bytes(dir / "missing-footer.jsonl", bytes.substr(0, footer));
    rejects([&] { Trajectory t(dir / "missing-footer.jsonl"); }, "missing footer accepted");
    std::vector<std::string> lines;
    for (std::size_t start = 0; start < bytes.size();) {
        const auto end = bytes.find('\n', start);
        require(end != std::string::npos, "unterminated generated line");
        lines.push_back(bytes.substr(start, end - start + 1));
        start = end + 1;
    }
    require(lines.size() == 5, "unexpected recording line count");
    const std::string reordered = lines[0] + lines[1] + lines[3] + lines[2];
    auto end_record = nlohmann::json::parse(lines[4]);
    end_record["sha256"] = sha256(reordered);
    put_bytes(dir / "backwards.jsonl", reordered + end_record.dump() + "\n");
    rejects([&] { Trajectory t(dir / "backwards.jsonl"); }, "checksummed out-of-order frames accepted");

    {
        TrajectoryRecorder recorder(path, "left", "new-serial", "new-firmware");
        rejects([&] { TrajectoryRecorder conflict(path, "right", "serial", "firmware"); }, "concurrent writer accepted");
        recorder.record(frame(100, 2.0, true));
        recorder.record(frame(300, 3.0, false));
        recorder.finish();
    }
    Trajectory replacement(path);
    require(replacement.hand() == "left" && replacement.serial() == "new-serial" && replacement.duration_ns() == 200,
            "existing file was not overwritten");
    near(replacement.at(0)[0], 2.0);
    {
        TrajectoryRecorder recorder(path, "right", "serial", "firmware");
        recorder.record(first);
        rejects([&] { recorder.record(frame(999, 0.0, false)); }, "backwards source accepted");
        rejects([&] { recorder.finish(); }, "failed recording committed");
    }
    rejects([&] { Trajectory t(path); }, "backwards recording replayable");
    {
        TrajectoryRecorder recorder(path, "right", "serial", "firmware");
        recorder.record(first); recorder.record(second);
        recorder.abort();
    }
    rejects([&] { Trajectory t(path); }, "aborted recording replayable");
}
} // namespace
int main() {
    try { TempDir dir; test(dir.path); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
