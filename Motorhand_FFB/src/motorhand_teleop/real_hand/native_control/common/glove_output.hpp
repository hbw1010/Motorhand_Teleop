#pragma once
#include "common/hand.hpp"

#include "common/feedback.hpp"
#include "common/jacobian_worker.hpp"
#include "types.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace motorhand::native {
class EncosDriver;
template<std::size_t N> class CommonDriverImpedancePlanner;

enum class GloveOutputMode { position, force };
template<std::size_t N> struct CommonGloveOutputResult {
    GloveOutputMode mode{};
    std::uint64_t request_id{};
    std::uint64_t source_ns{};
    std::uint64_t jacobian_source_ns{}; // Original Jacobian reference sample, never completion time.
    std::uint64_t last_write_ns{};
    Vec12 applied{}; // Quantized nominal model-axis value submitted to serial, not measured torque.
    FeedbackResult<N> force{};
    bool force_computed{};
    HandState<N> feedback_input_hand{}; // 产生本次输出的原始从手反馈，不是UI发布时的最新帧。
    JointArray<N> feedback_input_target{};
    bool driver_impedance{};
    DriverImpedanceSubmission impedance{};
    CommonDriverImpedanceObservation<N> impedance_observation{};
    std::uint64_t contact_evaluated_ns{}, impedance_planned_ns{};
};

// One output thread owns command computation/submission. Main control never
// waits for the 12 serial slots on each tick. Only the latest not-yet-started
// request is retained; errors latch. Every request keeps its original deadline.
template<std::size_t N> class CommonGloveOutputWorker {
public:
    CommonGloveOutputWorker(EncosDriver& driver, HandFeedback<N>* force, CommonDriverImpedancePlanner<N>* impedance = nullptr);
    ~CommonGloveOutputWorker();
    CommonGloveOutputWorker(const CommonGloveOutputWorker<N>&) = delete;
    CommonGloveOutputWorker<N>& operator=(const CommonGloveOutputWorker<N>&) = delete;

    std::uint64_t submit_position(const Vec12& motor_rad, std::uint64_t source_ns,
                                  std::uint64_t deadline_ns);
    std::uint64_t submit_force(const JointArray<N>& target, const HandState<N>& hand,
                               const GloveFrame& glove, const GlovePose& pose,
                               const std::optional<CommonJacobianSnapshot<N>>& jacobian,
                               std::uint64_t deadline_ns);
    std::optional<CommonGloveOutputResult<N>> poll();

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
