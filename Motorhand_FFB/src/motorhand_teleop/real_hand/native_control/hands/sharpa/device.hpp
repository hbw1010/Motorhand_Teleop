#pragma once
#include "transport.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>
namespace motorhand::native::sharpa {
struct StopStatus {
  bool expired{}, pending{}, confirmed{}, sdk_busy{};
};
class Device final : public HandDevice<22> {
public:
  Device(DeviceOptions, HandDescriptor<22>,
         std::unique_ptr<Transport> transport = {});
  ~Device() noexcept override;
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void connect() override;
  HandState<22> read() override;
  const HandDescriptor<22> &descriptor() const override { return descriptor_; }
  void enable() override;
  void home(const std::atomic<bool> *stop_requested = nullptr) override;
  void send(const JointArray<22> &,
            std::uint64_t original_deadline_ns) override;
  JointArray<22> submit_position(const JointArray<22> &,
                                 std::uint64_t original_deadline_ns) override;
  void disable() override;
  void close() override;
  StopStatus stop_status() const noexcept;

private:
  void ready_locked();
  Packet read_packet_locked();
  HandState<22> read_locked();
  void disable_locked();
  void fail_locked();
  void wait_enabled_locked(bool);
  void arm_locked(std::uint64_t source_ns, std::uint64_t received_ns);
  void watchdog();
  void request_stop();
  void check_permission();
  struct Safety {
    std::atomic<std::uint64_t> original_deadline{}, steady_deadline{};
    std::atomic<bool> expired{}, pending{}, confirmed{}, owned{}, sdk_busy{};
    std::atomic<bool> worker_started{}, worker_done{};
  };
  DeviceOptions options_;
  HandDescriptor<22> descriptor_;
  std::shared_ptr<Transport> transport_;
  std::shared_ptr<std::mutex> mutex_ = std::make_shared<std::mutex>();
  std::shared_ptr<Safety> safety_ = std::make_shared<Safety>();
  std::mutex monitor_mutex_;
  std::condition_variable cv_;
  std::thread watchdog_;
  bool connected_{}, closed_{}, faulted_{}, owned_{}, active_{}, homed_{},
      have_packet_{}, reported_cache_conflict_{};
  std::atomic<bool> stop_watchdog_{};
  std::int64_t clock_offset_{};
  Packet last_packet_;
  std::uint64_t received_ns_{}, deadline_ns_{}, last_send_ns_{};
  JointArray<22> last_target_{};
};
} // namespace motorhand::native::sharpa
