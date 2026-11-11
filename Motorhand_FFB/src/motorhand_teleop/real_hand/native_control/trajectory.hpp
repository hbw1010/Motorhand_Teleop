#pragma once

#include "types.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace motorhand::native {

struct TrajectoryFrame {
    std::uint64_t source_ns{};
    Vec20 position_rad{}, velocity_rad_s{}, effort_a{}, target_rad{};
    bool has_velocity{}, has_effort{};
};

class Trajectory {
public:
    explicit Trajectory(const std::filesystem::path& path);
    std::uint64_t duration_ns() const;
    Vec20 at(std::uint64_t elapsed_ns) const;
    void validate_limits(const Vec20& lower, const Vec20& upper) const;
    std::string hand() const;
    std::string serial() const;
    std::string firmware() const;
private:
    std::string hand_, serial_, firmware_;
    std::vector<std::uint64_t> times_;
    std::vector<Vec20> positions_;
};

class TrajectoryRecorder {
public:
    TrajectoryRecorder(const std::filesystem::path& path, std::string hand,
                       std::string serial, std::string firmware);
    ~TrajectoryRecorder();
    TrajectoryRecorder(const TrajectoryRecorder&) = delete;
    TrajectoryRecorder& operator=(const TrajectoryRecorder&) = delete;
    void record(const TrajectoryFrame& frame);
    void finish();
    void abort();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace motorhand::native
