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
    TelemetryPublisher(int port, const std::filesystem::path& glove_mapping);
    ~TelemetryPublisher();
    TelemetryPublisher(const TelemetryPublisher&) = delete;
    TelemetryPublisher& operator=(const TelemetryPublisher&) = delete;
    void publish(const SessionTick& tick) noexcept;
    std::uint64_t dropped_packets() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace motorhand::native
