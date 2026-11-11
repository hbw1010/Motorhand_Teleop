#include "glove_output.hpp"
#include "encos_driver.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace motorhand::native {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void check_source(std::uint64_t stamp, std::uint64_t now) {
    if (!stamp || stamp > now || now - stamp > 100000000ULL)
        throw std::runtime_error("Glove output source missing, future, or older than 100 ms");
}
void check_deadline(std::uint64_t deadline, std::uint64_t now) {
    if (now >= deadline) throw std::runtime_error("Glove output request deadline expired");
}
}

struct GloveOutputWorker::Impl {
    enum class Kind { position, force, enter_force, enter_position };
    struct Request {
        Kind kind{};
        std::uint64_t id{}, source{}, deadline{};
        Vec12 position{};
        Vec20 target{};
        HandFrame hand{};
        GloveFrame glove{};
        GlovePose pose{};
        std::optional<JacobianSnapshot> jacobian;
    };
    EncosDriver& driver;
    ForceController* force;
    std::mutex mutex;
    std::mutex close_mutex;
    std::condition_variable changed;
    std::optional<Request> pending;
    std::optional<GloveOutputResult> completed;
    std::exception_ptr failure;
    std::thread thread;
    GloveOutputMode mode = GloveOutputMode::position;
    std::uint64_t next_id = 0;
    bool barrier = false, closing = false, closed = false;

    Impl(EncosDriver& d, ForceController* f) : driver(d), force(f) {
        thread = std::thread([this] { run(); });
    }

    void check_locked() const {
        if (failure) std::rethrow_exception(failure);
        if (closing || closed) throw std::runtime_error("Glove output worker is closed");
    }
    std::uint64_t enqueue(Request&& req, GloveOutputMode expected) {
        std::lock_guard<std::mutex> lock(mutex);
        check_locked();
        if (barrier || mode != expected) throw std::runtime_error("Glove output mode transition or mode mismatch");
        req.id = ++next_id;
        const auto id = req.id;
        pending = std::move(req);
        changed.notify_one();
        return id;
    }
    void transition(Kind kind, std::uint64_t deadline = 0) {
        std::unique_lock<std::mutex> lock(mutex);
        check_locked();
        if (barrier) throw std::runtime_error("Glove output mode transition already in progress");
        if (kind == Kind::enter_force && !force) throw std::runtime_error("Glove force controller is not configured");
        barrier = true;
        completed.reset();
        Request req; req.kind = kind; req.deadline = deadline;
        pending = std::move(req); // Discard only work that has not started.
        changed.notify_one();
        changed.wait(lock, [&] { return !barrier || failure || closing || closed; });
        check_locked();
    }
    void fail(std::exception_ptr cause) noexcept {
        // The output thread is the sole driver writer. Cancel output immediately;
        // disable verifies zero-current serial submission and may itself fail.
        try {
            driver.disable();
        } catch (const std::exception& e) {
            try {
                try { std::rethrow_exception(cause); }
                catch (const std::exception& original) {
                    throw std::runtime_error(std::string("Glove output failed: ") + original.what() +
                                             "; emergency disable failed: " + e.what());
                }
                catch (...) {
                    throw std::runtime_error(std::string("Glove output failed; emergency disable failed: ") + e.what());
                }
            } catch (...) { cause = std::current_exception(); }
        } catch (...) {
            try { throw std::runtime_error("Glove output failed; emergency disable failed with unknown error"); }
            catch (...) { cause = std::current_exception(); }
        }
        std::lock_guard<std::mutex> lock(mutex);
        failure = cause; pending.reset(); completed.reset(); barrier = false;
        changed.notify_all();
    }
    void run() noexcept {
        try {
            for (;;) {
                Request req;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return closing || pending.has_value(); });
                    if (closing) break;
                    req = std::move(*pending);
                    pending.reset();
                }
                if (req.kind == Kind::enter_force) {
                    force->reset(now_ns());
                    driver.enable_force();
                    std::lock_guard<std::mutex> lock(mutex);
                    mode = GloveOutputMode::force; completed.reset(); barrier = false; changed.notify_all();
                    continue;
                }
                if (req.kind == Kind::enter_position) {
                    if (mode == GloveOutputMode::force) {
                        check_deadline(req.deadline, now_ns());
                        driver.write_torque(Vec12{}, req.deadline);
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    mode = GloveOutputMode::position; completed.reset(); barrier = false; changed.notify_all();
                    continue;
                }
                auto now = now_ns();
                check_deadline(req.deadline, now);
                check_source(req.source, now);
                if (req.deadline - req.source > 100000000ULL)
                    throw std::runtime_error("Glove output deadline exceeds source freshness window");
                GloveOutputResult result;
                result.request_id = req.id; result.source_ns = req.source;
                if (req.kind == Kind::position) {
                    result.mode = GloveOutputMode::position;
                    result.applied = driver.write_position(req.position, req.deadline);
                } else {
                    result.mode = GloveOutputMode::force;
                    check_source(req.hand.received_ns, now);
                    for (auto stamp : req.glove.received_ns) check_source(stamp, now);
                    // Zero source marks unavailable geometry. ForceController validates
                    // all real inputs and returns waiting_geometry with zero torque.
                    static constexpr std::array<double, 240> no_jacobian{};
                    const auto& jac = req.jacobian ? req.jacobian->jacobian : no_jacobian;
                    const auto& reference = req.jacobian ? req.jacobian->reference_joint_q12 : req.pose.joint_rad;
                    const auto jac_ns = req.jacobian ? req.jacobian->source_ns : 0;
                    result.force = force->step(req.target, req.hand, req.glove, req.pose, jac, reference, jac_ns, now);
                    result.jacobian_source_ns = jac_ns;
                    result.force_computed = req.jacobian.has_value();
                    now = now_ns();
                    check_deadline(req.deadline, now);
                    check_source(req.source, now);
                    result.applied = driver.write_torque(result.force.model_motor_nm, req.deadline);
                    force->acknowledge_torque(result.applied);
                }
                const auto stamps = driver.last_command_write_ns();
                result.last_write_ns = *std::max_element(stamps.begin(), stamps.end());
                if (!result.last_write_ns || result.last_write_ns >= req.deadline)
                    throw std::runtime_error("Glove output has no timely serial submission confirmation");
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    // A transition invalidates any completion that predates it.
                    if (!barrier && !closing) completed = result;
                }
            }
        } catch (...) { fail(std::current_exception()); }
        std::lock_guard<std::mutex> lock(mutex);
        closed = true; changed.notify_all();
    }
};

GloveOutputWorker::GloveOutputWorker(EncosDriver& driver, ForceController* force)
    : impl_(std::make_unique<Impl>(driver, force)) {}
GloveOutputWorker::~GloveOutputWorker() { try { close(); } catch (...) {} }

std::uint64_t GloveOutputWorker::submit_position(const Vec12& motor_rad, std::uint64_t source_ns,
                                                 std::uint64_t deadline_ns) {
    Impl::Request req; req.kind = Impl::Kind::position; req.position = motor_rad;
    req.source = source_ns; req.deadline = deadline_ns;
    return impl_->enqueue(std::move(req), GloveOutputMode::position);
}
std::uint64_t GloveOutputWorker::submit_force(const Vec20& target, const HandFrame& hand,
                                              const GloveFrame& glove, const GlovePose& pose,
                                              const std::optional<JacobianSnapshot>& jacobian,
                                              std::uint64_t deadline_ns) {
    Impl::Request req; req.kind = Impl::Kind::force; req.target = target; req.hand = hand;
    req.glove = glove; req.pose = pose; req.jacobian = jacobian;
    req.source = std::min(hand.received_ns, glove.oldest_ns); req.deadline = deadline_ns;
    return impl_->enqueue(std::move(req), GloveOutputMode::force);
}
std::optional<GloveOutputResult> GloveOutputWorker::poll() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check_locked();
    auto result = std::move(impl_->completed);
    impl_->completed.reset();
    return result;
}
void GloveOutputWorker::enter_force() { impl_->transition(Impl::Kind::enter_force); }
void GloveOutputWorker::enter_position(std::uint64_t deadline_ns) { impl_->transition(Impl::Kind::enter_position, deadline_ns); }
void GloveOutputWorker::close() {
    auto& d = *impl_;
    std::lock_guard<std::mutex> close_lock(d.close_mutex);
    {
        std::lock_guard<std::mutex> lock(d.mutex);
        d.closing = true; d.pending.reset(); d.completed.reset(); d.changed.notify_all();
    }
    if (d.thread.joinable()) d.thread.join();
    std::lock_guard<std::mutex> lock(d.mutex);
    if (d.failure) std::rethrow_exception(d.failure);
}
} // namespace motorhand::native
