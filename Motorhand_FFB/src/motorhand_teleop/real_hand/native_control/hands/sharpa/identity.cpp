#include "identity.hpp"
#include <stdexcept>
namespace motorhand::native::sharpa {
void verify_identity(const DeviceInfo &i, const DeviceOptions &o,
                     bool approve) {
  if (o.serial.empty() || i.serial != o.serial || !i.is_hand ||
      i.left != o.left || i.pn.empty() || i.firmware.empty() ||
      (!o.expected_pn.empty() && o.expected_pn != i.pn) ||
      (!o.approved_firmware.empty() && o.approved_firmware != i.firmware) ||
      (!o.address.empty() && o.address != i.address))
    throw std::runtime_error("Sharpa SN/PN/固件/HAND/手侧不匹配");
  if (approve && (o.expected_pn.empty() || o.approved_firmware.empty()))
    throw std::runtime_error("Sharpa使能必须明确批准PN及固件");
}
} // namespace motorhand::native::sharpa
