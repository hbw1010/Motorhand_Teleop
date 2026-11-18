#pragma once
#include "common/hand.hpp"
#include <string>
namespace motorhand::native::sharpa {
inline constexpr const char *torque_source = "sharpa-sdk-reported-joint-torque";
struct DeviceInfo {
  std::string serial, pn, firmware, address;
  bool left{}, is_hand = true;
  bool sdk_source{}, position_mode{};
};
struct DeviceOptions {
  std::string serial, expected_pn, approved_firmware, address;
  bool left{};
  double enable_timeout_s = 2, home_timeout_s = 8, transition_duration_s = 1,
         home_tolerance_rad = .04, max_velocity_rad_s = 3;
};
void verify_identity(const DeviceInfo &, const DeviceOptions &,
                     bool require_approval = false);
} // namespace motorhand::native::sharpa
