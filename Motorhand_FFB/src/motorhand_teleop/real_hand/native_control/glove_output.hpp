#pragma once

#include "force_controller.hpp"
#include "jacobian_worker.hpp"
#include "types.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace motorhand::native {
class EncosDriver;
class DriverImpedancePlanner;

enum class GloveOutputMode { position, force };
struct GloveOutputResult {
    GloveOutputMode mode{};
    std::uint64_t request_id{};
    std::uint64_t source_ns{};
    std::uint64_t jacobian_source_ns{}; // Original Jacobian reference sample, never completion time.
    std::uint64_t last_write_ns{};
    Vec12 applied{}; // Quantized nominal model-axis value submitted to serial, not measured torque.
    ForceResult force{};
    bool force_computed{};
    HandFrame feedback_input_hand{}; // 产生本次输出的原始从手反馈，不是UI发布时的最新帧。
    Vec20 feedback_input_target{};
    bool driver_impedance{};
    DriverImpedanceSubmission impedance{};
    DriverImpedanceObservation impedance_observation{};
    std::uint64_t contact_evaluated_ns{}, impedance_planned_ns{};
};

// One output thread owns command computation/submission. Main control never
// waits for the 12 serial slots on each tick. Only the latest not-yet-started
// request is retained; errors latch. Every request keeps its original deadline.
class GloveOutputWorker {
public:
    GloveOutputWorker(EncosDriver& driver, ForceController* force, DriverImpedancePlanner* impedance = nullptr);
    ~GloveOutputWorker();
    GloveOutputWorker(const GloveOutputWorker&) = delete;
    GloveOutputWorker& operator=(const GloveOutputWorker&) = delete;

    std::uint64_t submit_position(const Vec12& motor_rad, std::uint64_t source_ns,
                                  std::uint64_t deadline_ns);
    std::uint64_t submit_force(const Vec20& target, const HandFrame& hand,
                               const GloveFrame& glove, const GlovePose& pose,
                               const std::optional<JacobianSnapshot>& jacobian,
                               std::uint64_t deadline_ns);
    std::optional<GloveOutputResult> poll();

    // Mode transitions are infrequent barriers. They discard unstarted work,
    // wait for in-flight submission, and either enter force at zero history or
    // submit a zero force frame before allowing position commands.
    void enter_force();
    void enter_position(std::uint64_t deadline_ns);
    void close(); // Joins, reports worker errors; driver disable/close remains caller-owned.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace motorhand::native
