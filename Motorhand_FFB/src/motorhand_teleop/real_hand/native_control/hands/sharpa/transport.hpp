#pragma once
#include "identity.hpp"
#include <memory>
#include <vector>
namespace motorhand::native::sharpa {
struct Packet {
  std::uint64_t timestamp{}, host_recv_time_us{};
  std::uint16_t sequence{};
  std::vector<double> angles, velocities, torques;
};
// 只表达官方SDK接口；构造不得发现、连接或输出，工厂也不接触硬件。
class Transport {
public:
  virtual ~Transport() = default;
  virtual std::vector<DeviceInfo> discover() = 0;
  virtual void connect(const std::string &serial, bool disable_tactile,
                       bool disable_sync_time) = 0;
  virtual void start() = 0;
  virtual DeviceInfo device_info() = 0;
  virtual bool is_enabled() = 0;
  virtual bool has_fault() = 0;
  // 仅返回最近一次has_fault查询的缓存，不额外访问SDK。
  virtual std::string fault_details() const { return {}; }
  virtual Packet state() = 0;
  virtual void use_sdk_source() = 0;
  virtual void use_position_mode() = 0;
  virtual void set_enabled(bool) = 0;
  virtual void set_positions(const JointArray<22> &) = 0;
  virtual void release(const std::string &serial) = 0;
  virtual std::uint64_t monotonic_ns();
  virtual std::uint64_t system_ns();
  virtual void sleep_ns(std::uint64_t);
};
std::unique_ptr<Transport> make_sdk_transport();
} // namespace motorhand::native::sharpa
