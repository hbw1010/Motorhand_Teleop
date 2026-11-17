#include "wuji_identity.hpp"

#include <wuji_sdk.h>

#include <arpa/inet.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace motorhand::native {
namespace {

void check(WujiStatus status, const char* operation) {
    if (status == WUJI_STATUS_OK) return;
    const char* detail = wuji_last_error();
    throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "SDK error"));
}

struct SdkSession {
    SdkSession() {
        const char* version = wuji_version();
        if (!version || std::strcmp(version, "2026.8.31") != 0)
            throw std::runtime_error("需要官方 C SDK 2026.8.31");
        check(wuji_init(nullptr), "初始化 Wuji SDK");
    }
    ~SdkSession() { wuji_shutdown(); }
};
struct Device {
    WujiDevice* handle{};
    ~Device() {
        if (handle) { wuji_dev_disconnect(handle); wuji_dev_release(handle); }
    }
    void close() {
        check(wuji_dev_disconnect(handle), "断开 Wuji 只读预检");
        wuji_dev_release(handle);
        handle = nullptr;
    }
};

// Linux route lookup; no interface or network configuration is changed.
std::string route_interface(const std::string& address) {
    std::string host = address;
    const auto scheme = host.find("://");
    if (scheme != std::string::npos) host.erase(0, scheme + 3);
    const auto colon = host.find(':');
    if (colon != std::string::npos) host.resize(colon);
    in_addr ip{};
    if (inet_pton(AF_INET, host.c_str(), &ip) != 1) return {};
    const int fd = socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE);
    if (fd < 0) return {};
    struct Request { nlmsghdr header; rtmsg route; std::array<char, RTA_SPACE(4)> data; } request{};
    request.header.nlmsg_len = NLMSG_LENGTH(sizeof(rtmsg));
    request.header.nlmsg_type = RTM_GETROUTE;
    request.header.nlmsg_flags = NLM_F_REQUEST;
    request.header.nlmsg_seq = 1;
    request.route.rtm_family = AF_INET;
    request.route.rtm_dst_len = 32;
    auto* attr = reinterpret_cast<rtattr*>(reinterpret_cast<char*>(&request) + NLMSG_ALIGN(request.header.nlmsg_len));
    attr->rta_type = RTA_DST;
    attr->rta_len = RTA_LENGTH(sizeof(ip));
    std::memcpy(RTA_DATA(attr), &ip, sizeof(ip));
    request.header.nlmsg_len = NLMSG_ALIGN(request.header.nlmsg_len) + RTA_LENGTH(sizeof(ip));
    sockaddr_nl kernel{};
    kernel.nl_family = AF_NETLINK;
    if (sendto(fd, &request, request.header.nlmsg_len, 0,
               reinterpret_cast<const sockaddr*>(&kernel), sizeof(kernel)) < 0) {
        close(fd);
        return {};
    }
    timeval timeout{1, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    std::array<char, 8192> response{};
    ssize_t size = recv(fd, response.data(), response.size(), 0);
    close(fd);
    if (size <= 0) return {};
    int remaining = static_cast<int>(size);
    for (auto* h = reinterpret_cast<nlmsghdr*>(response.data()); NLMSG_OK(h, remaining); h = NLMSG_NEXT(h, remaining)) {
        if (h->nlmsg_seq != 1 || h->nlmsg_type != RTM_NEWROUTE) continue;
        auto* route = reinterpret_cast<rtmsg*>(NLMSG_DATA(h));
        int attributes = RTM_PAYLOAD(h);
        for (auto* a = RTM_RTA(route); RTA_OK(a, attributes); a = RTA_NEXT(a, attributes)) {
            if (a->rta_type != RTA_OIF || RTA_PAYLOAD(a) < sizeof(int)) continue;
            int index = 0;
            std::memcpy(&index, RTA_DATA(a), sizeof(index));
            std::array<char, IF_NAMESIZE> name{};
            if (index > 0 && if_indextoname(index, name.data())) return name.data();
        }
    }
    return {};
}
bool wired_route(const std::string& address) {
    const auto name = route_interface(address);
    if (name.empty() || name.find('/') != std::string::npos) return false;
    const auto base = std::filesystem::path("/sys/class/net") / name;
    std::ifstream type(base / "type");
    int value = 0;
    return (type >> value) && value == 1 && std::filesystem::exists(base / "device") &&
           !std::filesystem::exists(base / "wireless") && !std::filesystem::exists(base / "phy80211");
}

std::string actual_firmware(WujiDevice* device) {
    std::uint8_t* raw = nullptr;
    std::size_t size = 0;
    check(wuji_dev_get(device, "firmware_version", &raw, &size), "读取 Wuji 固件资源");
    std::vector<std::uint8_t> bytes;
    if (raw && size) bytes.assign(raw, raw + size);
    wuji_buffer_free(raw, size);
    // Fixed SDK 2026.8.31 reports Hand2 firmware_version as a length-prefixed
    // ASCII string. Unknown encodings fail closed, never inherit the approved value.
    if (bytes.size() < 7 || bytes.size() > 128 || bytes[0] != bytes.size() - 1 || bytes[1] != 'v')
        throw std::runtime_error("Wuji 固件资源编码未知");
    std::string result(bytes.begin() + 2, bytes.end());
    int dots = 0;
    bool digit = false;
    for (char c : result) {
        if (c >= '0' && c <= '9') digit = true;
        else if (c == '.' && digit) { ++dots; digit = false; }
        else throw std::runtime_error("Wuji 固件版本格式未知");
    }
    if (!digit || dots != 2) throw std::runtime_error("Wuji 固件版本格式未知");
    return result;
}

} // namespace

WujiIdentity inspect_wuji(const std::string& serial, bool left, std::uint32_t timeout_ms) {
    if (timeout_ms == 0 || serial.find('\0') != std::string::npos)
        throw std::invalid_argument("Wuji 预检参数非法");
    SdkSession sdk;
    WujiDiscovered* devices = nullptr;
    std::size_t count = 0;
    check(wuji_scan(&devices, &count), "发现 Wuji 设备");
    std::map<std::string, std::set<std::string>> candidates;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& d = devices[i];
        // SDK 2026.8.31 can report an IP endpoint with Zenoh transport.
        // Select by the concrete wired IP route below, not by transport label.
        if (d.device_id != WUJI_DEVICE_TYPE_WUJI_HAND_2 ||
            !d.serial_number[0] || !d.address[0] || (!serial.empty() && serial != d.serial_number)) continue;
        candidates[d.serial_number].insert(d.address);
    }
    wuji_discovered_free(devices, count);
    if (candidates.size() != 1) throw std::runtime_error("未发现唯一匹配的 Wuji Hand2；请检查连接或指定序列号");
    const auto& candidate = *candidates.begin();
    std::set<std::string> wired;
    for (const auto& address : candidate.second) if (wired_route(address)) wired.insert(address);
    if (wired.size() != 1) throw std::runtime_error("未发现唯一的物理有线 Wuji 路由，拒绝猜测网卡");
    WujiIdentity identity{candidate.first, *wired.begin(), {}, left};
    auto options = wuji_connect_options_default();
    options.timeout_ms = timeout_ms;
    options.retry_count = 0;
    options.enable_bridge = false;
    options.auto_time_sync_interval_enabled = false;
    WujiConnectTarget target{WUJI_CONNECT_TARGET_KIND_ADDR, identity.address.c_str()};
    Device device;
    check(wuji_connect(&target, "motorhand_native_preflight", &options, &device.handle), "连接 Wuji 只读预检");
    if (!device.handle || !wuji_dev_is_connected(device.handle)) throw std::runtime_error("Wuji 未连接");
    std::array<char, 128> actual_serial{};
    const int32_t serial_size = wuji_dev_serial_number(device.handle, actual_serial.data(), actual_serial.size());
    if (serial_size <= 0 || serial_size >= static_cast<int32_t>(actual_serial.size()) || identity.serial != actual_serial.data())
        throw std::runtime_error("Wuji 预检序列号不匹配");
    WujiHandedness side{};
    check(wuji_hand_2_get_handedness(device.handle, &side), "读取 Wuji 二代手侧");
    const auto expected = left ? WUJI_HANDEDNESS_LEFT : WUJI_HANDEDNESS_RIGHT;
    if (side != expected) throw std::runtime_error("Wuji 预检手侧不匹配");
    identity.firmware = actual_firmware(device.handle);
    device.close();
    return identity;
}

} // namespace motorhand::native
