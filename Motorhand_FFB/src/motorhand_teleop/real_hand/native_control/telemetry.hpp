#pragma once

#include "session.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace motorhand::native {

// Single control-thread producer. Sampling is a bounded copy only; JSON and
// localhost UDP transmission happen on a worker. Telemetry never controls a device.
class TelemetryPublisher {
public:
    TelemetryPublisher(int port, const std::filesystem::path& glove_mapping, bool left = false);
    ~TelemetryPublisher();
    TelemetryPublisher(const TelemetryPublisher&) = delete;
    TelemetryPublisher& operator=(const TelemetryPublisher&) = delete;
    void publish(const SessionTick& tick) noexcept;
    // 生命周期末尾的尽力通知；丢包时监视器仍依据时效显示断流。
    void status(const std::string& state, const std::string& message) noexcept;
    std::uint64_t dropped_packets() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace motorhand::native
