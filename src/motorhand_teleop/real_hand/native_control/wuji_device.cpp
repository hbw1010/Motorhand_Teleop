#include "wuji_device.hpp"

#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

namespace motorhand::native {
namespace {
constexpr std::size_t error_size = 2048;
using Error = std::array<char, error_size>;

void require_text(const std::string& text, const char* name) {
    if (text.empty() || text.find('\0') != std::string::npos ||
        std::isspace(static_cast<unsigned char>(text.front())) ||
        std::isspace(static_cast<unsigned char>(text.back())))
        throw std::invalid_argument(std::string("invalid Wuji ") + name);
}

std::string failure(const char* operation, int status, const Error& error) {
    return std::string("Wuji ") + operation + " failed (" + std::to_string(status) + "): " +
        (error[0] ? error.data() : "native ABI returned no error text");
}

void check(const char* operation, int status, const Error& error) {
    if (status != 0) throw std::runtime_error(failure(operation, status, error));
}
}  // namespace

WujiDevice::WujiDevice(const WujiIdentity& identity, const std::string& approved_firmware) {
    require_text(identity.serial, "serial");
    require_text(identity.address, "address");
    require_text(identity.firmware, "preflight firmware");
    require_text(approved_firmware, "approved firmware");
    if (identity.firmware != approved_firmware)
        throw std::invalid_argument("Wuji measured firmware does not match approved firmware: measured=" +
                                    identity.firmware + ", approved=" + approved_firmware);
    const int version = real_hand_abi_version();
    if (version != 6) throw std::runtime_error("Wuji native ABI mismatch: expected 6, got " + std::to_string(version));
    RealHandOptions options{};
    options.model = 2;
    options.hand = identity.left ? 0 : 1;
    options.serial_number = identity.serial.c_str();
    options.address = identity.address.c_str();
    options.approved_firmware_version = approved_firmware.c_str();
    options.preflight_firmware_version = identity.firmware.c_str();
    options.max_velocity = 0;
    options.max_acceleration = 0;
    options.realtime_max_velocity = 0;
    options.command_timeout = 0.5;
    options.feedback_timeout = 0.5;
    options.home_rate_hz = 100.0;
    options.home_tolerance = 0.15;
    options.home_timeout = 15.0;
    options.enable_timeout = 5.0;
    options.transition_duration = 1.5;
    options.effort_limit = -1.0;
    options.mit_kp = 5.0;
    options.mit_kd = 0.1;
    options.timeout_ms = 1000;
    options.auto_home = 0;
    Error error{};
    hand_ = real_hand_create(&options, error.data(), error.size());
    if (!hand_) throw std::runtime_error(failure("create", -1, error));
}

WujiDevice::~WujiDevice() noexcept {
    if (hand_) real_hand_destroy(hand_);
}

void WujiDevice::connect() {
    Error error{};
    const int status = real_hand_connect(hand_, error.data(), error.size());
    check("connect", status, error);
}

RealHandState WujiDevice::read() {
    RealHandState state{};
    Error error{};
    const int status = real_hand_read_state(hand_, &state, error.data(), error.size());
    check("read", status, error);
    return state;
}

void WujiDevice::enable() {
    Error error{};
    const int status = real_hand_enable(hand_, error.data(), error.size());
    check("enable", status, error);
}

void WujiDevice::send(const Vec20& positions_rad, std::uint64_t original_deadline_ns) {
    Error error{};
    const int status = real_hand_send(hand_, positions_rad.data(), positions_rad.size(),
                                      original_deadline_ns, error.data(), error.size());
    check("send", status, error);
}

void WujiDevice::disable() {
    Error error{};
    const int status = real_hand_disable(hand_, error.data(), error.size());
    check("disable", status, error);
}

void WujiDevice::close() {
    if (!close_error_.empty()) throw std::runtime_error(close_error_);
    if (closed_) return;
    Error error{};
    const int status = real_hand_close(hand_, error.data(), error.size());
    if (status != 0) {
        close_error_ = failure("close", status, error);
        throw std::runtime_error(close_error_);
    }
    closed_ = true;
}

}  // namespace motorhand::native
