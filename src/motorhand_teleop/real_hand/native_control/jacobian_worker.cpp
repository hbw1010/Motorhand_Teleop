#include "jacobian_worker.hpp"
#include "force_jacobian.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace motorhand::native {
namespace {
template <class Array> bool finite(const Array& values) {
    return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
}

struct JacobianWorker::Impl {
    struct Request {
        GlovePose pose;
        std::uint64_t source_ns;
        std::uint64_t id;
    };
    const std::filesystem::path asset_root, yaml_path;
    const bool left;
    const double step;
    std::mutex mutex;
    std::condition_variable changed;
    std::array<std::thread, 4> threads;
    bool stopping = false;
    std::exception_ptr failure;
    std::uint64_t next_id = 0;
    std::optional<Request> current, pending;
    std::optional<JacobianSnapshot> ready;
    std::array<std::array<double, 60>, 4> parts{};
    std::array<bool, 4> finished{};
    std::size_t count = 0;

    Impl(const std::filesystem::path& assets, const std::filesystem::path& yaml,
         bool is_left, double difference_step)
        : asset_root(assets), yaml_path(yaml), left(is_left), step(difference_step) {
        if (!std::isfinite(step) || step <= 0) throw std::invalid_argument("invalid Jacobian difference step");
        try {
            for (std::size_t i = 0; i < threads.size(); ++i)
                threads[i] = std::thread([this, i] { run(i); });
        } catch (...) {
            shutdown();
            throw;
        }
    }
    ~Impl() { shutdown(); }
    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        for (auto& thread : threads) if (thread.joinable()) thread.join();
    }
    void rethrow_failure() const {
        if (failure) std::rethrow_exception(failure);
        if (stopping) throw std::runtime_error("Jacobian worker stopped");
    }
    void launch(const Request& request) {
        current = request;
        finished.fill(false);
        count = 0;
        changed.notify_all();
    }
    void run(std::size_t index) noexcept {
        try {
            ForceJacobianColumns calculator(asset_root, yaml_path, left);
            const std::vector<int> columns{static_cast<int>(index * 3),
                                           static_cast<int>(index * 3 + 1),
                                           static_cast<int>(index * 3 + 2)};
            std::uint64_t previous = 0;
            for (;;) {
                Request work;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return stopping || (current && current->id != previous); });
                    if (stopping) return;
                    work = *current;
                }
                // All four calculations use the identical validated centre/assembly branch.
                const auto values = calculator.compute(work.pose, columns, step);
                if (values.size() != 60 || !finite(values))
                    throw std::runtime_error("invalid Jacobian worker result");
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (stopping) return;
                    if (!current || current->id != work.id || current->source_ns != work.source_ns ||
                        current->pose.joint_rad != work.pose.joint_rad || finished[index])
                        throw std::runtime_error("mismatched Jacobian worker result");
                    std::copy(values.begin(), values.end(), parts[index].begin());
                    finished[index] = true;
                    previous = work.id;
                    if (++count == 4) {
                        JacobianSnapshot result;
                        result.reference_joint_q12 = work.pose.joint_rad;
                        result.source_ns = work.source_ns;
                        result.request_id = work.id;
                        for (std::size_t row = 0; row < 20; ++row)
                            for (std::size_t part = 0; part < 4; ++part)
                                for (std::size_t col = 0; col < 3; ++col)
                                    result.jacobian[row * 12 + part * 3 + col] = parts[part][row * 3 + col];
                        ready = result;
                        if (pending) {
                            const Request next = *pending;
                            pending.reset();
                            launch(next);
                        } else {
                            current.reset();
                        }
                    }
                }
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure) failure = std::current_exception();
                stopping = true;
                ready.reset();
                pending.reset();
            }
            changed.notify_all();
        }
    }
};

JacobianWorker::JacobianWorker(const std::filesystem::path& asset_root,
                               const std::filesystem::path& yaml_path, bool left, double step_rad)
    : impl_(std::make_unique<Impl>(asset_root, yaml_path, left, step_rad)) {}
JacobianWorker::~JacobianWorker() = default;
std::uint64_t JacobianWorker::submit(const GlovePose& pose, std::uint64_t source_ns) {
    if (!source_ns || !finite(pose.joint_rad) || !finite(pose.motor_rad) || !finite(pose.tree_rad) ||
        !finite(pose.points_m) || !finite(pose.point_jacobian) || !finite(pose.motor_jacobian) ||
        !std::isfinite(pose.max_gap_m)) throw std::invalid_argument("invalid Jacobian source pose or time");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->rethrow_failure();
    if (impl_->next_id == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Jacobian request ID exhausted");
    const Impl::Request request{pose, source_ns, ++impl_->next_id};
    if (impl_->current) impl_->pending = request;
    else impl_->launch(request);
    return request.id;
}
std::optional<JacobianSnapshot> JacobianWorker::poll() {
    std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return std::nullopt;
    impl_->rethrow_failure();
    auto result = std::move(impl_->ready);
    impl_->ready.reset();
    return result;
}
} // namespace motorhand::native
