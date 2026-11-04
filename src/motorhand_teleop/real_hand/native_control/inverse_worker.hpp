#pragma once

#include "inverse_retarget.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>

namespace motorhand::native {

struct InverseWorkerResult {
    InverseResult result;
    std::uint64_t source_ns{};
    std::uint64_t request_id{};
};

// A worker exclusively owns its models and solver on its background thread.
// Not-yet-started requests and successful completions are latest-wins. A
// failed result or exception remains latched until polled. A completed request
// need not be the latest submitted: the caller must enforce original source
// and command deadlines, and avoid issuing out-of-order request IDs.
class InverseWorker {
public:
    InverseWorker(std::filesystem::path glove_asset_root,
                  std::filesystem::path hand_yaml_path, bool left,
                  std::array<std::array<double, 2>, 12> motor_bounds);
    ~InverseWorker();
    InverseWorker(const InverseWorker&) = delete;
    InverseWorker& operator=(const InverseWorker&) = delete;

    std::uint64_t submit(const HandFrame& hand, const GlovePose& seed,
                         std::uint64_t source_ns);
    // Non-blocking: nullopt means no completed request. Worker exceptions are
    // rethrown here; an unsuccessful InverseResult is returned, not thrown away.
    std::optional<InverseWorkerResult> poll();

private:
    struct Request {
        HandFrame hand;
        GlovePose seed;
        std::uint64_t source_ns;
        std::uint64_t request_id;
    };
    struct Completion {
        std::optional<InverseWorkerResult> value;
        std::exception_ptr error;
    };

    void run() noexcept;

    std::filesystem::path glove_asset_root_;
    std::filesystem::path hand_yaml_path_;
    bool left_;
    std::array<std::array<double, 2>, 12> motor_bounds_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<Request> pending_;
    std::optional<Completion> completed_;
    std::exception_ptr initialization_error_;
    std::uint64_t last_request_id_{};
    bool stopping_{};
    std::thread thread_;
};

}  // namespace motorhand::native
