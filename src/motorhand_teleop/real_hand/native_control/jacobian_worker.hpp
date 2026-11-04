#pragma once

#include "types.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

namespace motorhand::native {

struct JacobianSnapshot {
    std::array<double, 240> jacobian{}; // 20 x 12, row-major
    Vec12 reference_joint_q12{};
    std::uint64_t source_ns{}; // Original glove receive time, never completion time.
    std::uint64_t request_id{};
};

// Four persistent threads each own a separate CAD model and retarget optimizer.
// Thread-safe submit/poll; latest pending request replaces earlier pending work.
class JacobianWorker {
public:
    JacobianWorker(const std::filesystem::path& asset_root,
                   const std::filesystem::path& yaml_path, bool left, double step_rad);
    ~JacobianWorker();
    JacobianWorker(const JacobianWorker&) = delete;
    JacobianWorker& operator=(const JacobianWorker&) = delete;

    // pose must be the already validated complete CAD pose for source_ns.
    // A return value uniquely identifies this request (not a completion timestamp).
    std::uint64_t submit(const GlovePose& pose, std::uint64_t source_ns);
    // Nonblocking: consumes a complete snapshot if available; empty if busy or none.
    // Asynchronous initialization/calculation failures are rethrown here and by submit.
    std::optional<JacobianSnapshot> poll();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace motorhand::native
