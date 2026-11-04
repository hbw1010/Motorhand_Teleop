#include "inverse_worker.hpp"

#include "ve33_model.hpp"
#include "wuji_retarget.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace motorhand::native {

InverseWorker::InverseWorker(std::filesystem::path glove_asset_root,
                             std::filesystem::path hand_yaml_path, bool left,
                             std::array<std::array<double, 2>, 12> motor_bounds)
    : glove_asset_root_(std::move(glove_asset_root)),
      hand_yaml_path_(std::move(hand_yaml_path)), left_(left),
      motor_bounds_(motor_bounds), thread_(&InverseWorker::run, this) {}

InverseWorker::~InverseWorker() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        pending_.reset();
    }
    condition_.notify_one();
    if (thread_.joinable()) thread_.join();
}

std::uint64_t InverseWorker::submit(const HandFrame& hand, const GlovePose& seed,
                                    std::uint64_t source_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialization_error_) std::rethrow_exception(initialization_error_);
    if (stopping_) throw std::runtime_error("inverse worker is stopping");
    if (last_request_id_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("inverse worker request ID exhausted");
    const std::uint64_t id = ++last_request_id_;
    pending_ = Request{hand, seed, source_ns, id};
    condition_.notify_one();
    return id;
}

std::optional<InverseWorkerResult> InverseWorker::poll() {
    Completion completion;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!completed_) {
            if (initialization_error_) std::rethrow_exception(initialization_error_);
            return std::nullopt;
        }
        completion = std::move(*completed_);
        completed_.reset();
    }
    condition_.notify_one();
    if (completion.error) std::rethrow_exception(completion.error);
    return std::move(completion.value);
}

void InverseWorker::run() noexcept {
    try {
        // Construction and all model access happen on this thread alone. No
        // initialization or warmup pose is ever published as feedback.
        Ve33Model glove(glove_asset_root_, left_);
        WujiRetarget hand(hand_yaml_path_, left_);
        InverseRetarget inverse(glove, hand, motor_bounds_);
        for (;;) {
            std::optional<Request> request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    // Do not overwrite an unconsumed failure, but permit a
                    // newer successful result to replace an older success.
                    const bool failure_latched = completed_ &&
                        (completed_->error || !completed_->value->result.success);
                    return stopping_ || (pending_.has_value() && !failure_latched);
                });
                if (stopping_) return;
                request = std::move(pending_);
                pending_.reset();
            }

            Completion completion;
            try {
                InverseResult result = inverse.solve(request->hand.position_rad, request->seed);
                completion.value.emplace(InverseWorkerResult{
                    std::move(result), request->source_ns, request->request_id});
            } catch (...) {
                completion.error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) return;
                completed_ = std::move(completion);
            }
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        initialization_error_ = std::current_exception();
    }
}

}  // namespace motorhand::native
