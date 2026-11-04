// Offline-only link seam: compile this file with glove_output.cpp, without
// encos_driver.cpp or force_controller.cpp. These definitions never open a
// device; they observe ordering and inject failures, not physical behavior.
#include "../src/motorhand_teleop/real_hand/native_control/glove_output.hpp"
#include "../src/motorhand_teleop/real_hand/native_control/encos_driver.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace motorhand::native;
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F&& f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
struct Fake {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::string> events;
    std::vector<Vec12> position, torque;
    std::array<std::uint64_t, 12> stamps{};
    bool block = false, entered = false, fail = false;
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        require(cv.wait_for(lock, std::chrono::seconds(2), [&] { return entered; }), "worker did not enter fake driver");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex); block = false; cv.notify_all();
    }
    void reset() {
        std::lock_guard<std::mutex> lock(mutex);
        events.clear(); position.clear(); torque.clear(); stamps = {};
        block = entered = fail = false;
    }
} fake;
struct ReleaseOnExit { ~ReleaseOnExit() { fake.release(); } };
Vec12 write_fake(const Vec12& value, std::uint64_t deadline, bool torque) {
    std::unique_lock<std::mutex> lock(fake.mutex);
    fake.events.push_back(torque ? "torque" : "position");
    (torque ? fake.torque : fake.position).push_back(value);
    fake.entered = true; fake.cv.notify_all();
    fake.cv.wait(lock, [] { return !fake.block; });
    if (fake.fail) throw std::runtime_error("injected driver failure");
    if (now_ns() >= deadline) throw std::runtime_error("expired fake command");
    fake.stamps.fill(now_ns());
    Vec12 applied = value;
    for (double& x : applied) x += 0.001; // Simulated codec acknowledgement.
    return applied;
}
}

namespace motorhand::native {
struct EncosDriver::Impl {};
EncosDriver::EncosDriver(const std::string&, const std::string&) : impl_(std::make_unique<Impl>()) {}
EncosDriver::~EncosDriver() = default;
Vec12 EncosDriver::write_position(const Vec12& v, std::uint64_t d) { return write_fake(v, d, false); }
Vec12 EncosDriver::write_torque(const Vec12& v, std::uint64_t d) { return write_fake(v, d, true); }
void EncosDriver::enable_force() {
    std::lock_guard<std::mutex> lock(fake.mutex); fake.events.push_back("enable_force");
}
void EncosDriver::disable() {
    std::lock_guard<std::mutex> lock(fake.mutex); fake.events.push_back("disable");
}
std::array<std::uint64_t, 12> EncosDriver::last_command_write_ns() const {
    std::lock_guard<std::mutex> lock(fake.mutex); return fake.stamps;
}
ForceController::ForceController(const std::string&, const std::string&, const Vec12&, const Vec12&, const std::array<int, 12>&, bool) {}
void ForceController::reset(std::uint64_t) {
    std::lock_guard<std::mutex> lock(fake.mutex); fake.events.push_back("reset");
}
ForceResult ForceController::step(const Vec20&, const HandFrame&, const GloveFrame&, const GlovePose&,
                                  const std::array<double, 240>&, const Vec12&, std::uint64_t, std::uint64_t) {
    std::lock_guard<std::mutex> lock(fake.mutex);
    fake.events.push_back("compute");
    ForceResult result; result.active = true; result.model_motor_nm.fill(0.4); return result;
}
void ForceController::acknowledge_torque(const Vec12& actual) {
    std::lock_guard<std::mutex> lock(fake.mutex);
    require(std::abs(actual[0] - 0.401) < 1e-12, "force acknowledgement did not use applied command");
    fake.events.push_back("ack");
}
}

namespace {
Vec12 command(double value) { Vec12 result{}; result.fill(value); return result; }
std::optional<GloveOutputResult> completion(GloveOutputWorker& worker, std::uint64_t request) {
    const auto until = Clock::now() + std::chrono::seconds(2);
    while (Clock::now() < until) {
        auto result = worker.poll();
        if (result && result->request_id == request) return result;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return {};
}
void latest_test() {
    fake.reset(); EncosDriver driver("", ""); GloveOutputWorker worker(driver, nullptr);
    ReleaseOnExit release_on_exit;
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.block = true; }
    auto source = now_ns() - 3; const auto deadline = source + 99'000'000ULL;
    const auto first = worker.submit_position(command(1), source, deadline);
    fake.wait_entered();
    const auto start = Clock::now();
    worker.submit_position(command(2), source + 1, deadline);
    const auto last = worker.submit_position(command(3), source + 2, deadline);
    const auto submit_time = Clock::now() - start;
    fake.release();
    require(submit_time < std::chrono::milliseconds(100), "submit waited for serial output");
    const auto result = completion(worker, last);
    require(result.has_value(), "latest request not completed");
    require(first != last && result->source_ns == source + 2 && result->mode == GloveOutputMode::position,
            "request identity or source changed");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        require(fake.position.size() == 2 && fake.position[0][0] == 1 && fake.position[1][0] == 3,
                "pending request did not use latest-wins");
        require(result->last_write_ns == fake.stamps[0] && result->last_write_ns != 0,
                "completion timestamp is not original driver write time");
    }
    require(std::abs(result->applied[0] - 3.001) < 1e-12, "completion lacks applied value");
    worker.close();
}
void switch_and_force_test() {
    fake.reset(); EncosDriver driver("", ""); Vec12 v{}; std::array<int, 12> map{};
    ForceController force("", "", v, v, map); GloveOutputWorker worker(driver, &force);
    ReleaseOnExit release_on_exit;
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.block = true; }
    const auto source = now_ns(); const auto deadline = source + 99'000'000ULL;
    worker.submit_position(command(1), source, deadline); fake.wait_entered();
    worker.submit_position(command(2), source, deadline);
    auto barrier = std::async(std::launch::async, [&] { worker.enter_force(); });
    const bool waiting = barrier.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
    fake.release(); barrier.get();
    require(waiting, "mode switch did not wait for in-flight output");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        require(fake.position.size() == 1, "mode switch did not discard queued position");
    }
    HandFrame hand{}; hand.received_ns = now_ns();
    GloveFrame glove{}; glove.oldest_ns = hand.received_ns - 10;
    glove.received_ns.fill(glove.oldest_ns);
    GlovePose pose{}; JacobianSnapshot jac{}; jac.source_ns = glove.oldest_ns - 20;
    const auto force_id = worker.submit_force(Vec20{}, hand, glove, pose, jac, glove.oldest_ns + 99'000'000ULL);
    const auto result = completion(worker, force_id);
    require(result && result->force_computed && result->source_ns == glove.oldest_ns &&
            result->jacobian_source_ns == jac.source_ns && result->last_write_ns != 0 &&
            result->mode == GloveOutputMode::force && std::abs(result->applied[0] - 0.401) < 1e-12,
            "force result missing or source changed");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        const auto compute = std::find(fake.events.begin(), fake.events.end(), "compute");
        const auto send = std::find(fake.events.begin(), fake.events.end(), "torque");
        const auto ack = std::find(fake.events.begin(), fake.events.end(), "ack");
        require(compute != fake.events.end() && send != fake.events.end() && ack != fake.events.end() &&
                compute < send && send < ack, "force compute/send/ack order incorrect");
    }
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.entered = false; fake.block = true; }
    hand.received_ns = now_ns(); glove.oldest_ns = hand.received_ns;
    glove.received_ns.fill(glove.oldest_ns); jac.source_ns = glove.oldest_ns;
    worker.submit_force(Vec20{}, hand, glove, pose, jac, glove.oldest_ns + 99'000'000ULL);
    fake.wait_entered();
    worker.submit_force(Vec20{}, hand, glove, pose, jac, glove.oldest_ns + 99'000'000ULL);
    auto exit = std::async(std::launch::async, [&] { worker.enter_position(now_ns() + 5'000'000'000ULL); });
    const bool exit_waiting = exit.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
    fake.release(); exit.get();
    require(exit_waiting, "force exit did not wait for in-flight output");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        require(fake.torque.size() == 3 && std::all_of(fake.torque.back().begin(), fake.torque.back().end(),
                    [](double x) { return x == 0; }), "force mode exit did not send zero");
    }
    worker.close();
}
void failure_test(bool expired) {
    fake.reset(); EncosDriver driver("", ""); GloveOutputWorker worker(driver, nullptr);
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.fail = !expired; }
    const auto stamp = now_ns();
    try { worker.submit_position(command(1), stamp, expired ? stamp - 1 : stamp + 99'000'000ULL); }
    catch (const std::exception&) { /* synchronous rejection is also a safe failure */ }
    const auto until = Clock::now() + std::chrono::seconds(2);
    bool failed = false;
    while (Clock::now() < until && !failed) {
        try { worker.poll(); } catch (const std::exception&) { failed = true; }
        if (!failed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(failed, "worker failure was not latched");
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        require(std::find(fake.events.begin(), fake.events.end(), "disable") != fake.events.end(),
                "worker failure did not disable output");
    }
    rejects([&] { worker.submit_position(command(2), now_ns(), now_ns() + 5'000'000'000ULL); }, "failed worker accepted command");
    rejects([&] { worker.close(); }, "close did not report worker failure");
}
void close_test() {
    fake.reset(); EncosDriver driver("", ""); GloveOutputWorker worker(driver, nullptr);
    ReleaseOnExit release_on_exit;
    { std::lock_guard<std::mutex> lock(fake.mutex); fake.block = true; }
    const auto source = now_ns();
    worker.submit_position(command(1), source, source + 99'000'000ULL); fake.wait_entered();
    auto closing = std::async(std::launch::async, [&] { worker.close(); });
    const bool waiting = closing.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout;
    fake.release(); closing.get();
    require(waiting, "close did not join in-flight output");
}
}
int main() {
    try { latest_test(); switch_and_force_test(); failure_test(false); failure_test(true); close_test(); }
    catch (const std::exception& e) { fake.release(); std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
