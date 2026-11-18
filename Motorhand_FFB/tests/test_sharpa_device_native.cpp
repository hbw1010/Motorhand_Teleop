#include "hands/sharpa/device.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
using namespace motorhand::native;
using namespace motorhand::native::sharpa;
namespace {
struct Fake : Transport {
  DeviceInfo info{"test-sn", "WaveSE-R-01", "3.0.10", "", false, true};
  Packet packet;
  std::vector<std::string> writes;
  std::atomic<bool> enabled{};
  bool connected{}, mode_throws{}, auto_packet = true, fault{};
  std::uint64_t query_delay{};
  std::chrono::milliseconds mode_delay{};
  std::atomic<std::uint64_t> now{2'000'000'000}, system{100'000'000'000};
  Fake() {
    packet.angles.assign(22, .2);
    packet.velocities.assign(22, 0);
    packet.torques.assign(22, .1);
    fresh();
  }
  void fresh() {
    now += 2'000'000;
    system += 2'000'000;
    ++packet.timestamp;
    ++packet.sequence;
    packet.host_recv_time_us = system / 1000;
  }
  std::vector<DeviceInfo> discover() override { return {info}; }
  void connect(const std::string &sn, bool tactile, bool sync) override {
    assert(sn == info.serial && tactile && sync);
    connected = true;
    writes.push_back("connect");
  }
  void start() override { writes.push_back("start"); }
  DeviceInfo device_info() override { return info; }
  bool is_enabled() override { return enabled; }
  bool has_fault() override {
    now += query_delay;
    system += query_delay;
    return fault;
  }
  std::string fault_details() const { return "code=1281(0x501), joints=[0]"; }
  Packet state() override {
    if (auto_packet)
      fresh();
    return packet;
  }
  void use_sdk_source() override {
    assert(!enabled);
    writes.push_back("source");
    info.sdk_source = true;
  }
  void use_position_mode() override {
    writes.push_back("mode");
    std::this_thread::sleep_for(mode_delay);
    enabled = true;
    info.position_mode = true;
    if (mode_throws)
      throw std::runtime_error("mode failed after implicit enable");
  }
  void set_enabled(bool value) override {
    enabled = value;
    writes.push_back(value ? "enable" : "disable");
  }
  void set_positions(const JointArray<22> &q) override {
    writes.push_back("position");
    packet.angles.assign(q.begin(), q.end());
  }
  void release(const std::string &sn) override {
    assert(sn == info.serial);
    connected = false;
    writes.push_back("release");
  }
  std::uint64_t monotonic_ns() override { return now; }
  std::uint64_t system_ns() override { return system; }
  void sleep_ns(std::uint64_t dt) override {
    now += dt;
    system += dt;
  }
};
struct BlockingFake final : Fake {
  std::mutex block_mutex;
  std::condition_variable block_cv;
  bool block{}, entered{}, released{}, block_query{}, block_disable{};
  std::atomic<int> disable_requests{}, releases{}, targets{};
  void wait_blocked() {
    std::unique_lock<std::mutex> lock(block_mutex);
    if (block) {
      entered = true;
      block_cv.notify_all();
      block_cv.wait(lock, [&] { return released; });
    }
  }
  Packet state() override {
    if (!block_query)
      wait_blocked();
    return Fake::state();
  }
  DeviceInfo device_info() override {
    if (block_query)
      wait_blocked();
    return Fake::device_info();
  }
  void set_enabled(bool value) override {
    if (!value) {
      if (block_disable)
        wait_blocked();
      ++disable_requests;
      enabled = false;
    } else
      Fake::set_enabled(value);
  }
  void set_positions(const JointArray<22> &q) override {
    ++targets;
    Fake::set_positions(q);
  }
  void release(const std::string &sn) override {
    ++releases;
    Fake::release(sn);
  }
};

HandDescriptor<22> descriptor() {
  HandDescriptor<22> d;
  for (int i = 0; i < 22; ++i) {
    d.joint_names[i] = "joint" + std::to_string(i);
    d.lower_rad[i] = -1;
    d.upper_rad[i] = 1;
  }
  return d;
}
DeviceOptions options() {
  DeviceOptions o;
  o.serial = "test-sn";
  o.expected_pn = "WaveSE-R-01";
  o.approved_firmware = "3.0.10";
  o.transition_duration_s = .02;
  o.home_timeout_s = 1;
  return o;
}
template <class F> void rejects(F fn) {
  bool caught = false;
  try {
    fn();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
void blocking_sdk_cannot_hold_deadline_monitor() {
  for (bool query : {false, true}) {
    auto t = std::make_unique<BlockingFake>();
    auto *f = t.get();
    Device d(options(), descriptor(), std::move(t));
    d.connect();
    d.enable();
    d.home();
    {
      std::lock_guard<std::mutex> lock(f->block_mutex);
      f->block = true;
      f->block_query = query;
    }
    std::atomic<bool> rejected{}, returned{};
    auto old_targets = f->targets.load();
    std::thread business([&] {
      try {
        JointArray<22> q{};
        q[21] = .3;
        d.send(q, f->now + hand_timeout_ns);
      } catch (const std::exception &) {
        rejected = true;
      }
      returned = true;
    });
    {
      std::unique_lock<std::mutex> lock(f->block_mutex);
      assert(f->block_cv.wait_for(lock, std::chrono::seconds(1),
                                  [&] { return f->entered; }));
    }
    std::atomic<int> queued_rejected{};
    std::vector<std::thread> queued;
    for (int i = 0; i < 2; ++i)
      queued.emplace_back([&] {
        try {
          JointArray<22> q{};
          d.send(q, f->now + hand_timeout_ns);
        } catch (const std::exception &) {
          ++queued_rejected;
        }
      });
    // SDK真实阻塞，未推进替身时钟；监察不得等待业务锁，也不能并发调用SDK。
    std::this_thread::sleep_for(std::chrono::milliseconds(140));
    const auto status = d.stop_status();
    const bool revoked = status.expired && status.pending &&
                         !status.confirmed && status.sdk_busy;
    const bool serial_only = f->disable_requests == 0 && f->enabled;
    const bool still_blocked = !returned;
    const bool no_release = f->releases == 0;
    rejects([&] { d.close(); });
    {
      std::lock_guard<std::mutex> lock(f->block_mutex);
      f->released = true;
      f->block_cv.notify_all();
    }
    business.join();
    for (auto &thread : queued)
      thread.join();
    assert(revoked && serial_only && still_blocked && no_release);
    assert(rejected && queued_rejected == 2);
    assert(f->targets == old_targets);
    assert(f->disable_requests > 0 && !f->enabled);
    const auto stopped = d.stop_status();
    assert(stopped.expired && stopped.confirmed && !stopped.pending);
    rejects([&] { d.read(); });
    d.close();
    d.close();
    assert(f->releases == 1);
  }
}

void unconfirmed_disable_keeps_live_handle() {
  auto t = std::make_unique<BlockingFake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  {
    std::lock_guard<std::mutex> lock(f->block_mutex);
    f->block = true;
    f->block_disable = true;
  }
  f->now += hand_timeout_ns;
  f->system += hand_timeout_ns;
  {
    std::unique_lock<std::mutex> lock(f->block_mutex);
    assert(f->block_cv.wait_for(lock, std::chrono::seconds(1),
                                [&] { return f->entered; }));
  }
  const auto pending = d.stop_status();
  assert(pending.expired && pending.pending && !pending.confirmed &&
         pending.sdk_busy);
  rejects([&] { d.close(); });
  assert(f->releases == 0);
  {
    std::lock_guard<std::mutex> lock(f->block_mutex);
    f->released = true;
    f->block_cv.notify_all();
  }
  for (int i = 0; i < 200 && !d.stop_status().confirmed; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(d.stop_status().confirmed);
  d.close();
  d.close();
  assert(f->releases == 1);
}

void readonly_and_duplicate() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  auto a = d.read();
  assert(a.has_load && a.load[21] == .1);
  assert(a.load_unit == HandLoadUnit::newton_metre);
  assert(a.online_source == HandStatusSource::complete_state_packet);
  assert(a.enabled_source == HandStatusSource::device_report);
  assert(f->writes == std::vector<std::string>({"connect", "start"}));
  f->auto_packet = false;
  f->packet.angles[21] = .4;
  f->packet.torques[21] = .3;
  auto b = d.read();
  assert(b.received_ns == a.received_ns && b.position_rad == a.position_rad &&
         b.load == a.load);
  f->fresh();
  const auto c = d.read();
  assert(c.received_ns > a.received_ns && c.position_rad[21] == .4 &&
         c.load[21] == .3);
  f->packet.angles[21] = .6;
  f->now += 100'000'000;
  f->system += 100'000'000;
  rejects([&] { d.read(); });
  assert(!f->enabled);
  d.close();
  d.close();
  assert(f->writes.back() == "release");
}
void enable_and_home_order() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  assert((f->writes == std::vector<std::string>{"connect", "start", "source",
                                                "mode"}));
  rejects([&] {
    JointArray<22> q{};
    d.send(q, f->now + 100'000'000);
  });
  assert(!f->enabled);
}
void home_then_send() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  d.home();
  JointArray<22> q{};
  q[21] = .5;
  const auto submitted = d.submit_position(q, f->now + 100'000'000);
  assert(submitted[21] > 0 && submitted[21] < q[21]);
  assert(std::equal(submitted.begin(), submitted.end(), f->packet.angles.begin()));
  d.disable();
  d.close();
}
void configured_position_mode_does_not_repeat_slow_sdk_setup() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  f->info.sdk_source = true;
  f->info.position_mode = true;
  // SDK 5.0.11 的模式切换内部等待超过实时命令的 100 ms 期限。
  f->mode_delay = std::chrono::milliseconds(120);
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  assert((f->writes == std::vector<std::string>{"connect", "start", "enable"}));
  d.home();
  assert(f->enabled && !d.stop_status().expired);
  d.close();
}
void late_enable_confirmation_uses_remaining_budget() {
  struct DelayedEnable final : Fake {
    std::uint64_t requested{};
    bool is_enabled() override {
      if (requested && now - requested >= 94'000'000)
        enabled = true;
      return enabled;
    }
    void set_enabled(bool value) override {
      if (!value) {
        requested = 0;
        Fake::set_enabled(false);
        return;
      }
      requested = now;
      writes.push_back("enable");
      sleep_ns(80'000'000);
    }
  };
  auto t = std::make_unique<DelayedEnable>();
  auto *f = t.get();
  f->info.sdk_source = f->info.position_mode = true;
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  assert(f->enabled && f->now - f->requested < hand_timeout_ns);
  d.home();
  assert(!d.stop_status().expired);
  d.close();
}
void enable_confirmation_requires_fresh_packets() {
  struct DelayedStatus final : Fake {
    std::uint64_t requested{};
    bool freeze_feedback{};
    bool is_enabled() override {
      if (requested && now - requested >= 150'000'000)
        enabled = true;
      return enabled;
    }
    void set_enabled(bool value) override {
      if (!value) {
        requested = 0;
        Fake::set_enabled(false);
        return;
      }
      requested = now;
      writes.push_back("enable");
      auto_packet = !freeze_feedback;
      sleep_ns(80'000'000);
    }
  };
  for (bool freeze : {false, true}) {
    auto t = std::make_unique<DelayedStatus>();
    auto *f = t.get();
    f->info.sdk_source = f->info.position_mode = true;
    f->freeze_feedback = freeze;
    Device d(options(), descriptor(), std::move(t));
    d.connect();
    if (freeze) {
      rejects([&] { d.enable(); });
      assert(!f->enabled);
    } else {
      d.enable();
      assert(f->enabled && f->now - f->requested >= 150'000'000);
      assert(std::count(f->writes.begin(), f->writes.end(), "enable") == 1);
    }
    assert(std::count(f->writes.begin(), f->writes.end(), "position") == 0);
    d.close();
  }
}
void slow_mode_change_also_revokes_permission() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  f->info.sdk_source = true;
  f->mode_delay = std::chrono::milliseconds(120);
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  bool diagnosed = false;
  try {
    d.enable();
  } catch (const std::runtime_error &e) {
    const std::string message = e.what();
    diagnosed = message.find("切换SDK位置模式") != std::string::npos &&
                message.find("100ms") != std::string::npos;
  }
  assert(diagnosed && d.stop_status().expired && !f->enabled);
  assert(std::count(f->writes.begin(), f->writes.end(), "position") == 0);
  d.close();
}
void implicit_enable_failure() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  f->mode_throws = true;
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  rejects([&] { d.enable(); });
  assert(!f->enabled);
  assert(f->writes.back() == "disable");
}
void refuse_external_enable() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  f->enabled = true;
  Device d(options(), descriptor(), std::move(t));
  rejects([&] { d.connect(); });
  assert(f->enabled);
  assert(std::find(f->writes.begin(), f->writes.end(), "disable") ==
         f->writes.end());
}
void malformed_packets() {
  for (int kind = 0; kind < 5; ++kind) {
    auto t = std::make_unique<Fake>();
    auto *f = t.get();
    Device d(options(), descriptor(), std::move(t));
    d.connect();
    d.read();
    f->auto_packet = false;
    if (kind == 0)
      f->packet.angles.pop_back();
    if (kind == 1)
      f->packet.torques[21] = std::numeric_limits<double>::quiet_NaN();
    if (kind == 2)
      --f->packet.sequence;
    if (kind == 3)
      f->system += 51'000'000;
    if (kind == 4)
      f->packet.host_recv_time_us += 1000;
    rejects([&] { d.read(); });
  }
}
void missing_load_allows_position() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  f->packet.torques.clear();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  assert(!d.read().has_load);
}
void mismatch_identity() {
  auto t = std::make_unique<Fake>();
  t->info.left = true;
  Device d(options(), descriptor(), std::move(t));
  rejects([&] { d.connect(); });
}
void slow_sdk_query_cannot_return_stale_state() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  f->query_delay = 100'000'000;
  rejects([&] { d.read(); });
}
void watchdog_stops_without_more_calls() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  f->now += 100'000'000;
  f->system += 100'000'000;
  for (int i = 0; i < 50 && f->enabled; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  assert(!f->enabled);
  rejects([&] { d.read(); });
}
void original_deadline_no_position() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  d.home();
  auto count = std::count(f->writes.begin(), f->writes.end(), "position");
  rejects([&] {
    JointArray<22> q{};
    d.send(q, f->now);
  });
  assert(count == std::count(f->writes.begin(), f->writes.end(), "position"));
  assert(!f->enabled);
}
void fault_message_preserves_sdk_details() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  f->fault = true;
  bool caught = false;
  try {
    d.read();
  } catch (const std::runtime_error &e) {
    caught = true;
    const std::string text = e.what();
    assert(text.find("1281") != std::string::npos);
    assert(text.find("joints=[0]") != std::string::npos);
  }
  assert(caught);
  assert(!f->enabled);
}

void sequence_wrap_and_fault() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  f->packet.sequence = 65534;
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  assert(d.read().sequence == 65535);
  assert(d.read().sequence == 0);
  d.enable();
  f->fault = true;
  rejects([&] { d.read(); });
  assert(!f->enabled);
}
void home_requires_new_arrival_frame() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  d.enable();
  f->auto_packet = false;
  rejects([&] { d.home(); });
  assert(!f->enabled);
}
void read_clock_conversion_preserves_packet_age() {
  auto t = std::make_unique<Fake>();
  auto *f = t.get();
  Device d(options(), descriptor(), std::move(t));
  d.connect();
  f->auto_packet = false;
  f->now += 40'000'000;
  f->system += 40'000'000;
  auto s = d.read();
  assert(f->now - s.received_ns == 40'000'000);
}

} // namespace
int main() {
  configured_position_mode_does_not_repeat_slow_sdk_setup();
  late_enable_confirmation_uses_remaining_budget();
  enable_confirmation_requires_fresh_packets();
  slow_mode_change_also_revokes_permission();
  blocking_sdk_cannot_hold_deadline_monitor();
  unconfirmed_disable_keeps_live_handle();
  fault_message_preserves_sdk_details();
  slow_sdk_query_cannot_return_stale_state();
  watchdog_stops_without_more_calls();
  original_deadline_no_position();
  sequence_wrap_and_fault();
  home_requires_new_arrival_frame();
  read_clock_conversion_preserves_packet_age();
  readonly_and_duplicate();
  enable_and_home_order();
  home_then_send();
  implicit_enable_failure();
  refuse_external_enable();
  malformed_packets();
  missing_load_allows_position();
  mismatch_identity();
  std::cout << "Sharpa SDK替身生命周期通过\n";
}
