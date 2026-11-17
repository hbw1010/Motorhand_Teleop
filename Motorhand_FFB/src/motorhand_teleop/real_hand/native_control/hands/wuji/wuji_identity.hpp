#pragma once

#include <cstdint>
#include <string>

namespace motorhand::native {

struct WujiIdentity {
    std::string serial;
    std::string address;
    std::string firmware;
    bool left{};
};

// Read-only SDK discovery and identity preflight. No enable, homing or control.
// Multiple candidate hands or multiple wired routes are rejected, not guessed.
WujiIdentity inspect_wuji(const std::string& serial, bool left,
                          std::uint32_t timeout_ms = 1000);

} // namespace motorhand::native
