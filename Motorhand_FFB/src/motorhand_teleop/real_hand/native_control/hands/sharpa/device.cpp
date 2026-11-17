#include "device.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
namespace motorhand::native::sharpa {
namespace {
std::uint64_t seconds(double s) { return static_cast<std::uint64_t>(s * 1e9); }
void fresh(std::uint64_t stamp, std::uint64_t now) {
  if (!stamp || stamp > now || now - stamp >= hand_timeout_ns)
    throw std::runtime_error("Sharpa原始反馈超过100ms期限或来自未来");
}
void finite_packet(const std::vector<double> &q) {
  for (double x : q)
    if (!std::isfinite(x))
      throw std::runtime_error("Sharpa状态包含非有限值");
}
} // namespace
Device::Device(DeviceOptions o, HandDescriptor<22> d,
               std::unique_ptr<Transport> t)
    : options_(std::move(o)), descriptor_(std::move(d)),
      transport_(std::move(t)) {
  if (options_.serial.empty())
    throw std::invalid_argument("Sharpa必须显式指定SN");
  for (double v : {options_.enable_timeout_s, options_.home_timeout_s,
                   options_.transition_duration_s, options_.home_tolerance_rad,
                   options_.max_velocity_rad_s})
    if (!std::isfinite(v) || v <= 0 || v > 3600)
      throw std::invalid_argument("Sharpa生命周期参数非法");
  for (int i = 0; i < 22; ++i) {
    if (descriptor_.joint_names[i].empty() ||
        !std::isfinite(descriptor_.lower_rad[i]) ||
        !std::isfinite(descriptor_.upper_rad[i]) ||
        descriptor_.lower_rad[i] >= descriptor_.upper_rad[i])
      throw std::invalid_argument("Sharpa模型关节名或限位无效");
    for (int j = 0; j < i; ++j)
      if (descriptor_.joint_names[i] == descriptor_.joint_names[j])
        throw std::invalid_argument("Sharpa关节名重复");
  }
  descriptor_.online_source = HandStatusSource::complete_state_packet;
  descriptor_.enabled_source = HandStatusSource::device_report;
  descriptor_.load_unit = HandLoadUnit::newton_metre;
  descriptor_.load_source = torque_source;
  if (!transport_)
    transport_ = make_sdk_transport();
  watchdog_ = std::thread(&Device::watchdog, this);
}
Device::~Device() noexcept {
  try {
    close();
  } catch (...) {
  }
  if (watchdog_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_watchdog_ = true;
      cv_.notify_all();
    }
    watchdog_.join();
  }
}
void Device::ready_locked() {
  if (closed_ || !connected_ || faulted_)
    throw std::runtime_error("Sharpa未连接、已关闭或故障锁存");
  if (active_ && deadline_ns_ && transport_->monotonic_ns() >= deadline_ns_)
    throw std::runtime_error("Sharpa命令/反馈原始期限已过");
}
void Device::connect() {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    if (closed_ || faulted_)
      throw std::runtime_error("Sharpa已关闭或故障锁存");
    if (connected_)
      return;
    auto end = transport_->monotonic_ns() + seconds(options_.enable_timeout_s);
    DeviceInfo found;
    bool have = false;
    do {
      int count = 0;
      for (const auto &i : transport_->discover())
        if (i.serial == options_.serial) {
          found = i;
          ++count;
        }
      if (count > 1)
        throw std::runtime_error("Sharpa发现重复SN");
      if (count) {
        verify_identity(found, options_);
        have = true;
        break;
      }
      transport_->sleep_ns(10'000'000);
    } while (transport_->monotonic_ns() < end);
    if (!have)
      throw std::runtime_error("Sharpa未发现指定SN");
    transport_->connect(options_.serial, true, true);
    connected_ = true;
    verify_identity(transport_->device_info(), options_);
    if (transport_->is_enabled())
      throw std::runtime_error("Sharpa已被外部使能，拒绝接管");
    transport_->start();
    if (transport_->is_enabled())
      throw std::runtime_error("Sharpa连接期间被外部使能");
    clock_offset_ = static_cast<std::int64_t>(transport_->system_ns()) -
                    static_cast<std::int64_t>(transport_->monotonic_ns());
    const auto info = transport_->device_info();
    verify_identity(info, options_);
    descriptor_.identity = {"sharpa", info.serial, info.firmware, info.left};
  } catch (...) {
    fail_locked();
    if (connected_) {
      try {
        transport_->release(options_.serial);
        connected_ = false;
      } catch (...) {
      }
    }
    throw;
  }
}
HandState<22> Device::read_locked() {
  ready_locked();
  auto p = transport_->state();
  auto now = transport_->monotonic_ns();
  auto offset = static_cast<std::int64_t>(transport_->system_ns()) -
                static_cast<std::int64_t>(now);
  if (std::abs(offset - clock_offset_) > 50'000'000)
    throw std::runtime_error("Sharpa系统时钟跳变");
  if (p.angles.size() != 22 ||
      (!p.velocities.empty() && p.velocities.size() != 22) ||
      (!p.torques.empty() && p.torques.size() != 22))
    throw std::runtime_error("Sharpa状态不是完整22关节包");
  finite_packet(p.angles);
  finite_packet(p.velocities);
  finite_packet(p.torques);
  bool same = have_packet_ && p.timestamp == last_packet_.timestamp &&
              p.sequence == last_packet_.sequence &&
              p.host_recv_time_us == last_packet_.host_recv_time_us;
  if (!same) {
    if (!p.timestamp || !p.host_recv_time_us ||
        p.host_recv_time_us >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max() / 1000))
      throw std::runtime_error("Sharpa SDK原始时间缺失或越界");
    if (have_packet_) {
      auto delta =
          static_cast<std::uint16_t>(p.sequence - last_packet_.sequence);
      if (p.timestamp <= last_packet_.timestamp || !delta || delta >= 32768 ||
          p.host_recv_time_us <= last_packet_.host_recv_time_us)
        throw std::runtime_error("Sharpa状态乱序或部分重复");
    }
    const auto received =
        static_cast<std::int64_t>(p.host_recv_time_us * 1000) -
        std::max(offset, clock_offset_);
    if (received <= 0 || received > static_cast<std::int64_t>(now))
      throw std::runtime_error("Sharpa接收时间来自未来或无效");
    received_ns_ = received;
    last_packet_ = p;
    have_packet_ = true;
  } else if (p.angles != last_packet_.angles ||
             p.velocities != last_packet_.velocities ||
             p.torques != last_packet_.torques)
    throw std::runtime_error("Sharpa重复缓存内容发生变化");
  fresh(received_ns_, now);
  const auto info = transport_->device_info();
  verify_identity(info, options_);
  bool enabled = transport_->is_enabled();
  if (transport_->has_fault())
    throw std::runtime_error("Sharpa SDK报告报警: " +
                             transport_->fault_details());
  if (active_ && !enabled)
    throw std::runtime_error("Sharpa活动期间报告失能");
  if (!owned_ && enabled)
    throw std::runtime_error("Sharpa外部使能，拒绝接管");
  fresh(received_ns_, transport_->monotonic_ns());
  ready_locked();
  HandState<22> s;
  s.identity = {"sharpa", info.serial, info.firmware, info.left};
  s.connected = true;
  s.received_ns = received_ns_;
  s.device_time = p.timestamp;
  s.sequence = p.sequence;
  s.online.fill(true);
  s.enabled.fill(enabled);
  s.online_source = descriptor_.online_source;
  s.enabled_source = descriptor_.enabled_source;
  s.load_unit = descriptor_.load_unit;
  s.load_source = descriptor_.load_source;
  std::copy(p.angles.begin(), p.angles.end(), s.position_rad.begin());
  s.has_velocity = !p.velocities.empty();
  s.has_load = !p.torques.empty();
  if (s.has_velocity)
    std::copy(p.velocities.begin(), p.velocities.end(),
              s.velocity_rad_s.begin());
  if (s.has_load)
    std::copy(p.torques.begin(), p.torques.end(), s.load.begin());
  return s;
}
HandState<22> Device::read() {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    return read_locked();
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::wait_enabled_locked(bool desired) {
  auto end = transport_->monotonic_ns() + seconds(options_.enable_timeout_s);
  while (transport_->is_enabled() != desired) {
    if (transport_->monotonic_ns() >= end)
      throw std::runtime_error("Sharpa整手使能确认超时");
    transport_->sleep_ns(10'000'000);
  }
}
void Device::arm_locked(std::uint64_t source, const HandState<22> &s) {
  auto now = transport_->monotonic_ns();
  fresh(s.received_ns, now);
  if (source <= now)
    throw std::runtime_error("Sharpa原始命令期限已过");
  deadline_ns_ = std::min(
      {source, now + hand_timeout_ns, s.received_ns + hand_timeout_ns});
  cv_.notify_all();
}
void Device::enable() {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    ready_locked();
    if (active_)
      throw std::runtime_error("Sharpa重复使能");
    verify_identity(transport_->device_info(), options_, true);
    if (transport_->is_enabled())
      throw std::runtime_error("Sharpa外部已使能");
    read_locked();
    owned_ = true;
    transport_->use_sdk_source();
    read_locked();
    transport_->use_position_mode();
    transport_->set_enabled(true);
    wait_enabled_locked(true);
    auto s = read_locked();
    active_ = true;
    homed_ = false;
    arm_locked(transport_->monotonic_ns() + hand_timeout_ns, s);
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::home(const std::atomic<bool> *stop) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    ready_locked();
    if (!active_)
      throw std::runtime_error("Sharpa回零前必须使能");
    for (int i = 0; i < 22; ++i)
      if (descriptor_.lower_rad[i] > 0 || descriptor_.upper_rad[i] < 0)
        throw std::runtime_error("Sharpa模型零位超限");
    auto s = read_locked();
    JointArray<22> start{};
    for (int i = 0; i < 22; ++i)
      start[i] = std::clamp(s.position_rad[i], descriptor_.lower_rad[i],
                            descriptor_.upper_rad[i]);
    auto started = transport_->monotonic_ns(),
         end = started + seconds(options_.home_timeout_s);
    std::uint64_t issued = 0;
    while (true) {
      ready_locked();
      if (stop && stop->load())
        throw std::runtime_error("Sharpa回零被停止");
      auto now = transport_->monotonic_ns();
      if (now >= end)
        throw std::runtime_error("Sharpa回零超时");
      if (!issued) {
        double t = std::clamp(
            (now - started) * 1e-9 / options_.transition_duration_s, 0., 1.);
        double w = t * t * t * (10 + t * (-15 + 6 * t));
        JointArray<22> q;
        for (int i = 0; i < 22; ++i)
          q[i] = start[i] * (1 - w);
        arm_locked(end, s);
        transport_->set_positions(q);
        if (transport_->monotonic_ns() >= deadline_ns_)
          throw std::runtime_error("Sharpa回零发送跨越期限");
        if (t >= 1)
          issued = transport_->monotonic_ns();
      }
      s = read_locked();
      bool arrived = true;
      for (double q : s.position_rad)
        arrived &= std::abs(q) <= options_.home_tolerance_rad;
      if (issued && s.received_ns > issued && arrived) {
        homed_ = true;
        last_target_ = {};
        last_send_ns_ = transport_->monotonic_ns();
        arm_locked(last_send_ns_ + hand_timeout_ns, s);
        return;
      }
      arm_locked(end, s);
      transport_->sleep_ns(10'000'000);
    }
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::send(const JointArray<22> &q, std::uint64_t source) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    ready_locked();
    if (!active_ || !homed_)
      throw std::runtime_error("Sharpa必须显式使能并完成本实例回零");
    for (int i = 0; i < 22; ++i)
      if (!std::isfinite(q[i]) || q[i] < descriptor_.lower_rad[i] ||
          q[i] > descriptor_.upper_rad[i])
        throw std::runtime_error("Sharpa目标无效或超限");
    auto s = read_locked();
    auto now = transport_->monotonic_ns();
    JointArray<22> output;
    double step = options_.max_velocity_rad_s * (now - last_send_ns_) * 1e-9;
    for (int i = 0; i < 22; ++i)
      output[i] =
          last_target_[i] + std::clamp(q[i] - last_target_[i], -step, step);
    arm_locked(source, s);
    transport_->set_positions(output);
    if (transport_->monotonic_ns() >= deadline_ns_)
      throw std::runtime_error("Sharpa发送跨越原始期限");
    last_target_ = output;
    last_send_ns_ = now;
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::disable_locked() {
  active_ = false;
  homed_ = false;
  deadline_ns_ = 0;
  if (!connected_ || !owned_)
    return;
  transport_->set_enabled(false);
  wait_enabled_locked(false);
  owned_ = false;
}
void Device::fail_locked() {
  faulted_ = true;
  try {
    disable_locked();
  } catch (...) {
    throw std::runtime_error("Sharpa故障锁存且失能未确认，需现场检查");
  }
}
void Device::disable() {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    disable_locked();
  } catch (...) {
    faulted_ = true;
    throw;
  }
}
void Device::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_watchdog_ = true;
    cv_.notify_all();
  }
  if (watchdog_.joinable())
    watchdog_.join();
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_)
    return;
  std::exception_ptr failure;
  try {
    disable_locked();
  } catch (...) {
    faulted_ = true;
    failure = std::current_exception();
  }
  if (connected_) {
    try {
      transport_->release(options_.serial);
      connected_ = false;
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  }
  if (!connected_)
    closed_ = true;
  if (failure)
    std::rethrow_exception(failure);
}
void Device::watchdog() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_watchdog_) {
    cv_.wait_for(lock, std::chrono::milliseconds(2));
    if (stop_watchdog_)
      break;
    if (active_ && deadline_ns_ && transport_->monotonic_ns() >= deadline_ns_)
      try {
        fail_locked();
      } catch (...) {
        faulted_ = true;
      }
  }
}
} // namespace motorhand::native::sharpa
