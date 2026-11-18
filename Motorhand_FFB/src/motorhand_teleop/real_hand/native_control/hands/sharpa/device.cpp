#include "device.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace motorhand::native::sharpa {
namespace {
std::uint64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
struct Busy {
  std::atomic<bool> &busy;
  explicit Busy(std::atomic<bool> &value) : busy(value) { busy = true; }
  ~Busy() { busy = false; }
};
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
  stop_watchdog_ = true;
  cv_.notify_all();
  if (watchdog_.joinable())
    watchdog_.join();
  // 未完成的stop worker持有共享运输句柄与串行锁，不能销毁其SDK对象。
}
StopStatus Device::stop_status() const noexcept {
  return {safety_->expired.load(), safety_->pending.load(),
          safety_->confirmed.load(), safety_->sdk_busy.load()};
}
void Device::check_permission() {
  const auto deadline = safety_->steady_deadline.load();
  if (deadline && steady_ns() >= deadline) {
    safety_->expired = true;
    request_stop();
  }
  if (safety_->expired)
    throw std::runtime_error(
        "Sharpa原始100ms期限已过，发令许可已撤销；停止状态请检查stop_status");
}
void Device::ready_locked() {
  check_permission();
  if (closed_ || !connected_ || faulted_)
    throw std::runtime_error("Sharpa未连接、已关闭或故障锁存");
  if (active_ && deadline_ns_ && transport_->monotonic_ns() >= deadline_ns_) {
    safety_->expired = true;
    request_stop();
    throw std::runtime_error("Sharpa命令/反馈原始期限已过");
  }
}
void Device::connect() {
  std::lock_guard<std::mutex> lock(*mutex_);
  Busy busy(safety_->sdk_busy);
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
Packet Device::read_packet_locked() {
  ready_locked();
  auto p = transport_->state();
  check_permission();
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
             p.torques != last_packet_.torques) {
    // SDK 偶尔返回内容变化但包身份未变的快照。只用上一份完整数据，
    // 保留其原始收包时间，连续缓存冲突仍会因超时而停机。
    p = last_packet_;
    if (!reported_cache_conflict_) {
      std::clog << "Sharpa丢弃同一时间戳下内容不一致的SDK缓存，沿用上一帧原始时间\n";
      reported_cache_conflict_ = true;
    }
  }
  fresh(received_ns_, now);
  return p;
}
HandState<22> Device::read_locked() {
  const auto p = read_packet_locked();
  const auto info = transport_->device_info();
  check_permission();
  verify_identity(info, options_);
  bool enabled = transport_->is_enabled();
  check_permission();
  const bool fault = transport_->has_fault();
  check_permission();
  if (fault)
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
  std::lock_guard<std::mutex> lock(*mutex_);
  Busy busy(safety_->sdk_busy);
  try {
    return read_locked();
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::wait_enabled_locked(bool desired) {
  auto end = transport_->monotonic_ns() + seconds(options_.enable_timeout_s);
  while (true) {
    if (desired) {
      // 使能 RPC 已在原期限内返回。仅凭新鲜关节包维持确认阶段，
      // 整段等待有独立上限；此处没有位置输出，也不重发使能。
      read_packet_locked();
      arm_locked(end, received_ns_);
    }
    const bool enabled = transport_->is_enabled();
    if (desired)
      check_permission();
    if (enabled == desired)
      return;
    if (transport_->monotonic_ns() >= end)
      throw std::runtime_error("Sharpa整手使能确认超时");
    // 使能 RPC 返回时，100ms 预算可能已剩不足 20ms。
    transport_->sleep_ns(desired ? 2'000'000 : 10'000'000);
  }
}
void Device::arm_locked(std::uint64_t source, std::uint64_t received_ns) {
  check_permission();
  auto now = transport_->monotonic_ns();
  fresh(received_ns, now);
  if (source <= now)
    throw std::runtime_error("Sharpa原始命令期限已过");
  deadline_ns_ = std::min(
      {source, now + hand_timeout_ns, received_ns + hand_timeout_ns});
  safety_->steady_deadline = steady_ns() + (deadline_ns_ - now);
  safety_->original_deadline = deadline_ns_;
  cv_.notify_all();
}
void Device::enable() {
  std::lock_guard<std::mutex> lock(*mutex_);
  Busy busy(safety_->sdk_busy);
  const char *phase = "使能前检查";
  std::uint64_t armed_ns = 0, enable_return_ns = 0, feedback_age_ns = 0;
  try {
    ready_locked();
    if (active_)
      throw std::runtime_error("Sharpa重复使能");
    const auto info = transport_->device_info();
    verify_identity(info, options_, true);
    if (transport_->is_enabled())
      throw std::runtime_error("Sharpa外部已使能");
    const auto initial = read_locked();
    armed_ns = transport_->monotonic_ns();
    feedback_age_ns = armed_ns - initial.received_ns;
    arm_locked(armed_ns + hand_timeout_ns, initial.received_ns);
    owned_ = true;
    safety_->owned = true;
    safety_->confirmed = false;
    check_permission();
    if (!info.sdk_source) {
      phase = "配置SDK控制源";
      transport_->use_sdk_source();
      check_permission();
      read_locked();
    }
    phase = info.position_mode ? "SDK显式使能" : "切换SDK位置模式";
    if (info.position_mode) {
      transport_->set_enabled(true);
    } else {
      // set_control_mode 自带一次使能，已为该动作建立原始期限。
      transport_->use_position_mode();
    }
    enable_return_ns = transport_->monotonic_ns();
    check_permission();
    phase = "读取SDK使能确认";
    wait_enabled_locked(true);
    phase = "确认使能后读取状态";
    // 已取得使能确认。用新的原始关节包进入就绪状态；home 会在发目标前再验身份和故障。
    read_packet_locked();
    active_ = true;
    homed_ = false;
    arm_locked(transport_->monotonic_ns() + hand_timeout_ns, received_ns_);
  } catch (const std::exception &error) {
    const auto failed_ns = transport_->monotonic_ns();
    fail_locked();
    std::string diagnostic = std::string("Sharpa") + phase + "：" + error.what();
    if (armed_ns) {
      diagnostic += "（起始反馈龄 " + std::to_string(feedback_age_ns / 1e6) +
                    " ms，使能阶段累计 " + std::to_string((failed_ns - armed_ns) / 1e6) +
                    " ms";
      if (enable_return_ns)
        diagnostic += "，SDK使能调用返回 " + std::to_string((enable_return_ns - armed_ns) / 1e6) + " ms";
      diagnostic += "）";
    }
    throw std::runtime_error(diagnostic);
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::home(const std::atomic<bool> *stop) {
  std::lock_guard<std::mutex> lock(*mutex_);
  Busy busy(safety_->sdk_busy);
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
        arm_locked(end, s.received_ns);
        check_permission();
        transport_->set_positions(q);
        check_permission();
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
        arm_locked(last_send_ns_ + hand_timeout_ns, s.received_ns);
        return;
      }
      arm_locked(end, s.received_ns);
      transport_->sleep_ns(10'000'000);
    }
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::send(const JointArray<22> &q, std::uint64_t source) {
  submit_position(q, source);
}
JointArray<22> Device::submit_position(const JointArray<22> &q,
                                      std::uint64_t source) {
  std::lock_guard<std::mutex> lock(*mutex_);
  Busy busy(safety_->sdk_busy);
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
    for (int i = 0; i < 22; ++i) {
      const auto delta = q[i] - last_target_[i];
      output[i] = std::abs(delta) <= step ? q[i] :
          last_target_[i] + std::copysign(step, delta);
    }
    arm_locked(source, s.received_ns);
    check_permission();
    transport_->set_positions(output);
    check_permission();
    if (transport_->monotonic_ns() >= deadline_ns_)
      throw std::runtime_error("Sharpa发送跨越原始期限");
    last_target_ = output;
    last_send_ns_ = now;
    return output;
  } catch (...) {
    fail_locked();
    throw;
  }
}
void Device::disable_locked() {
  active_ = false;
  homed_ = false;
  deadline_ns_ = 0;
  safety_->original_deadline = 0;
  safety_->steady_deadline = 0;
  if (!connected_ || !owned_)
    return;
  if (!safety_->owned) {
    owned_ = false;
    return;
  }
  safety_->pending = true;
  safety_->confirmed = false;
  transport_->set_enabled(false);
  wait_enabled_locked(false);
  owned_ = false;
  safety_->owned = false;
  safety_->confirmed = true;
  safety_->pending = false;
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
  std::lock_guard<std::mutex> lock(*mutex_);
  Busy busy(safety_->sdk_busy);
  try {
    disable_locked();
  } catch (...) {
    faulted_ = true;
    throw;
  }
}
void Device::close() {
  {
    std::unique_lock<std::mutex> lock(*mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      throw std::runtime_error("Sharpa SDK忙，停止未确认；保留仍在使用的句柄");
    if (closed_)
      return;
    stop_watchdog_ = true;
    cv_.notify_all();
  }
  if (watchdog_.joinable())
    watchdog_.join();
  std::unique_lock<std::mutex> lock(*mutex_, std::try_to_lock);
  if (!lock.owns_lock())
    throw std::runtime_error("Sharpa SDK忙，停止未确认；保留仍在使用的句柄");
  Busy busy(safety_->sdk_busy);
  disable_locked();
  if (safety_->worker_started && !safety_->worker_done) {
    lock.unlock();
    const auto end = steady_ns() + hand_timeout_ns;
    while (!safety_->worker_done && steady_ns() < end)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!safety_->worker_done)
      throw std::runtime_error("Sharpa停止工作器尚未退出；保留SDK句柄");
    lock.lock();
    safety_->sdk_busy = true;
  }
  if (safety_->pending)
    throw std::runtime_error("Sharpa停止未确认；禁止销毁SDK句柄");
  if (connected_) {
    transport_->release(options_.serial);
    connected_ = false;
  }
  closed_ = true;
}
void Device::request_stop() {
  if (!safety_->owned)
    return;
  safety_->pending = true;
  safety_->confirmed = false;
  if (!safety_->owned) {
    safety_->confirmed = true;
    safety_->pending = false;
    return;
  }
  if (safety_->worker_started.exchange(true))
    return;
  auto state = safety_;
  auto transport = transport_;
  auto serial = mutex_;
  std::thread([state, transport, serial] {
    // SDK并发安全未获官方保证；失能仍通过同一串行通道排队。
    // 此工作器永不持有Device指针，阻塞时保留运输与串行锁的所有权。
    while (state->pending) {
      std::unique_lock<std::mutex> lock(*serial, std::try_to_lock);
      if (!lock.owns_lock()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      if (!state->pending || !state->owned) {
        if (!state->owned) {
          state->confirmed = true;
          state->pending = false;
        }
        break;
      }
      Busy busy(state->sdk_busy);
      try {
        transport->set_enabled(false);
        const auto end = steady_ns() + 2'000'000'000;
        while (transport->is_enabled()) {
          if (steady_ns() >= end)
            throw std::runtime_error("Sharpa失能确认超时");
          transport->sleep_ns(10'000'000);
        }
        state->owned = false;
        state->confirmed = true;
        state->pending = false;
      } catch (...) { /* pending保持true；不能把排队或异常当作失能确认。 */
      }
      break;
    }
    state->worker_done = true;
  }).detach();
}
void Device::watchdog() {
  std::unique_lock<std::mutex> lock(monitor_mutex_);
  while (!stop_watchdog_) {
    cv_.wait_for(lock, std::chrono::milliseconds(1));
    if (stop_watchdog_)
      break;
    const auto real = safety_->steady_deadline.load();
    const auto original = safety_->original_deadline.load();
    // 实际墙钟监察独立于SDK与业务锁；替身时钟也不能绕过原始期限。
    if (real && original &&
        (steady_ns() >= real || transport_->monotonic_ns() >= original)) {
      safety_->expired = true;
      request_stop();
    }
  }
}
} // namespace motorhand::native::sharpa
