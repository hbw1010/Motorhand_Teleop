#include "transport.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#ifdef MOTORHAND_WITH_SHARPA_SDK
#include <SharpaWaveSDK.h>
#include <mutex>
#endif
namespace motorhand::native::sharpa {
std::uint64_t Transport::monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
std::uint64_t Transport::system_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
void Transport::sleep_ns(std::uint64_t dt) {
  std::this_thread::sleep_for(std::chrono::nanoseconds(dt));
}
#ifdef MOTORHAND_WITH_SHARPA_SDK
namespace {
std::mutex manager_mutex;
void check(const ::sharpa::Error &e) {
  if (e.code)
    throw std::runtime_error("Sharpa SDK: " + e.message);
}
DeviceInfo convert(const ::sharpa::DeviceInfo &i) {
  return {i.sn,
          i.pn,
          i.firmware_version,
          i.ip,
          i.hand_side == ::sharpa::HandSide::LEFT,
          i.device_type == ::sharpa::DeviceType::HAND};
}
class SdkTransport final : public Transport {
  ::sharpa::SharpaWave *wave_{};
  std::string fault_details_;

public:
  std::vector<DeviceInfo> discover() override {
    std::lock_guard<std::mutex> lock(manager_mutex);
    std::vector<DeviceInfo> out;
    for (const auto &i :
         ::sharpa::SharpaWaveManager::get_instance().get_all_devices())
      out.push_back(convert(i));
    return out;
  }
  void connect(const std::string &sn, bool tactile, bool sync) override {
    std::lock_guard<std::mutex> lock(manager_mutex);
    auto &m = ::sharpa::SharpaWaveManager::get_instance();
    if (m.is_connected(sn))
      throw std::runtime_error("Sharpa SN已由外部实例连接");
    ::sharpa::SharpaWaveConfig c;
    c.disable_tactile = tactile;
    c.disable_sync_time = sync;
    wave_ = &m.connect(sn, c);
  }
  void start() override {
    if (!wave_->start())
      throw std::runtime_error("Sharpa start失败");
  }
  DeviceInfo device_info() override {
    return convert(wave_->get_device_info());
  }
  bool is_enabled() override {
    auto e = wave_->get_enable_state();
    check(e.first);
    return e.second;
  }
  bool has_fault() override {
    std::ostringstream text;
    bool fault = false;
    const auto faults = wave_->get_fault_code();
    for (const auto &entry : faults) {
      if (entry.first == ::sharpa::FaultCode::NORMAL)
        continue;
      if (fault)
        text << "; ";
      fault = true;
      const auto code = static_cast<int>(entry.first);
      text << "code=" << code << "(0x" << std::hex << std::setw(4)
           << std::setfill('0') << code << std::dec << "), joints=[";
      for (std::size_t index = 0; index < entry.second.size(); ++index) {
        if (index)
          text << ',';
        text << entry.second[index];
      }
      text << ']';
    }
    fault_details_ = text.str();
    return fault;
  }
  std::string fault_details() const override { return fault_details_; }
  Packet state() override {
    auto s = wave_->get_states();
    return {s.timestamp,
            s.host_recv_time_us,
            s.sequence,
            {s.angles.begin(), s.angles.end()},
            {s.velocities.begin(), s.velocities.end()},
            {s.torques.begin(), s.torques.end()}};
  }
  void use_sdk_source() override {
    check(wave_->set_control_source(::sharpa::ControlSource::SDK));
  }
  void use_position_mode() override {
    check(wave_->set_control_mode(::sharpa::ControlMode::POSITION));
  }
  void set_enabled(bool b) override { check(wave_->set_enable_state(b)); }
  void set_positions(const JointArray<22> &q) override {
    check(wave_->set_joint_position({q.begin(), q.end()}, false));
  }
  void release(const std::string &sn) override {
    if (!wave_)
      return;
    if (!wave_->stop())
      throw std::runtime_error("Sharpa stop失败，保留句柄");
    wave_->destroy();
    std::lock_guard<std::mutex> lock(manager_mutex);
    ::sharpa::SharpaWaveManager::get_instance().disconnect(sn);
    wave_ = nullptr;
  }
};
} // namespace
std::unique_ptr<Transport> make_sdk_transport() {
  return std::make_unique<SdkTransport>();
}
#else
std::unique_ptr<Transport> make_sdk_transport() {
  throw std::runtime_error("构建未启用Sharpa官方SDK");
}
#endif
} // namespace motorhand::native::sharpa
