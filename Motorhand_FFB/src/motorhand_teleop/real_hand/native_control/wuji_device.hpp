#pragma once

#include "../native/real_hand.h"
#include "types.hpp"
#include "wuji_identity.hpp"

#include <cstdint>
#include <atomic>
#include <string>

namespace motorhand::native {

// Construction and connect never enable the hand. Explicit close reports cleanup
// errors; destruction only performs the C ABI's best-effort safe release.
class WujiDevice {
public:
    WujiDevice(const WujiIdentity& identity, const std::string& approved_firmware);
    ~WujiDevice() noexcept;
    WujiDevice(const WujiDevice&) = delete;
    WujiDevice& operator=(const WujiDevice&) = delete;
    WujiDevice(WujiDevice&&) = delete;
    WujiDevice& operator=(WujiDevice&&) = delete;

    void connect();
    RealHandState read();
    void enable();
    void home(const std::atomic<bool>* stop_requested = nullptr);
    void send(const Vec20& positions_rad, std::uint64_t original_deadline_ns);
    void disable();
    void close();

private:
    RealHand* hand_ = nullptr;
    bool closed_ = false;
    std::string close_error_;
};

}  // namespace motorhand::native
