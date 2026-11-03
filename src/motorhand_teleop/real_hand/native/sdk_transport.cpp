#include "transport.h"
#include <wuji_sdk.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace real_hand {
namespace {
void check(WujiStatus status, const char *operation) {
    if (status != WUJI_STATUS_OK) {
        const char *detail = wuji_last_error();
        throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "SDK未提供错误详情"));
    }
}
constexpr const char *fingers[] = {"thumb", "index", "middle", "ring", "pinky"};
class SDKAdapter final : public Transport {
    WujiDevice *device = nullptr;
    WujiSub *states = nullptr, *diagnostics = nullptr;
    WujiJointCommandPublisher *publisher = nullptr;
    std::mutex feedback_mutex;
    Feedback latest;
    std::string stream_error;
    std::atomic<bool> closing{false};
    std::array<int, 256> indices{};

    template<class Entry> bool order(const Entry *entries, size_t count, std::array<int, 20> &out) {
        if (!entries || count != 20) return false;
        uint32_t seen = 0;
        for (size_t i = 0; i < 20; ++i) {
            const int index = indices[entries[i].nid];
            if (index < 0 || (seen & (1u << index))) return false;
            seen |= 1u << index;
            out[index] = static_cast<int>(i);
        }
        return seen == 0xfffff;
    }
    template<class Function> void callback(WujiFrameKind kind, Function &&fn) noexcept {
        try {
            std::lock_guard<std::mutex> lock(feedback_mutex);
            if (closing.load()) return;
            if (kind != WUJI_FRAME_KIND_OK) {
                stream_error = "SDK反馈流中断、丢帧或解码失败";
                return;
            }
            fn();
        } catch (...) {
            // 不让异常穿过SDK工作线程的C回调。
            closing.store(true);
        }
    }
    static void on_states(WujiFrameKind kind, const WujiJointStateFrame *frame, void *data) noexcept {
        auto &self = *static_cast<SDKAdapter *>(data);
        self.callback(kind, [&] {
            std::array<int, 20> ordered{};
            if (!frame || frame->num_joints != 20 || !self.order(frame->joints, frame->joints_len, ordered)) {
                self.stream_error = "位置反馈必须包含20个语义匹配且唯一的NID";
                return;
            }
            Angles positions, velocities, effort;
            bool has_velocity = true, has_effort = true;
            for (size_t i = 0; i < 20; ++i) {
                positions[i] = frame->joints[ordered[i]].position;
                velocities[i] = frame->joints[ordered[i]].velocity;
                effort[i] = frame->joints[ordered[i]].effort;
                has_velocity = has_velocity && std::isfinite(velocities[i]);
                has_effort = has_effort && std::isfinite(effort[i]);
                if (!std::isfinite(positions[i])) {
                    self.stream_error = "位置反馈含非有限值";
                    return;
                }
            }
            self.latest.positions = positions;
            // 可选量缺失/非有限只撤销该量的完整帧可用性，不伪零或沿用旧帧。
            self.latest.velocities_rad_s = velocities;
            self.latest.effort_a = effort;
            self.latest.has_velocity = has_velocity;
            self.latest.has_effort = has_effort;
            self.latest.position_ns = monotonic_ns();
            self.latest.device_us = frame->header.timestamp_us;
            self.latest.complete = self.latest.diagnostics_ns != 0;
        });
    }
    static void on_diagnostics(WujiFrameKind kind, const WujiJointDiagnosticsFrame *frame, void *data) noexcept {
        auto &self = *static_cast<SDKAdapter *>(data);
        self.callback(kind, [&] {
            std::array<int, 20> ordered{};
            if (!frame || frame->num_joints != 20 || !self.order(frame->joints, frame->joints_len, ordered)) {
                self.stream_error = "诊断反馈必须包含20个语义匹配且唯一的NID";
                return;
            }
            std::array<uint8_t, 20> enabled{};
            std::array<uint32_t, 20> errors{};
            std::array<bool, 20> warnings{};
            for (size_t i = 0; i < 20; ++i) {
                const auto &entry = frame->joints[ordered[i]];
                WujiJointStatus status{};
                check(wuji_hand_2_decode_joint_status(entry.status_word, &status), "解码关节状态");
                if (status.ext_state > 3 || status.enabled != (status.ext_state == 2)) {
                    self.stream_error = "诊断报告未知或矛盾的使能状态";
                    return;
                }
                enabled[i] = status.enabled;
                errors[i] = entry.error_code_current;
                if (errors[i] != 0) {
                    if (errors[i] == self.latest.errors[i]) {
                        warnings[i] = self.latest.warnings[i];
                    } else {
                        WujiErrorInfo info{};
                        warnings[i] = wuji_hand_2_describe_error(entry.error_code_current, &info) == WUJI_STATUS_OK &&
                                      std::strncmp(info.severity, "Warning", sizeof(info.severity)) == 0;
                    }
                }
            }
            self.latest.enabled = enabled;
            self.latest.errors = errors;
            self.latest.warnings = warnings;
            self.latest.online.fill(1); // 仅20个唯一在线诊断条目通过核验后成立。
            self.latest.diagnostics_ns = monotonic_ns();
            self.latest.complete = self.latest.position_ns != 0;
        });
    }
    void close_publisher() noexcept {
        if (publisher) { wuji_joint_command_publisher_close(publisher); publisher = nullptr; }
    }
public:
    ~SDKAdapter() override { try { release(); } catch (...) {} }
    Identity connect(const RealHandOptions &options) override {
        const char *version = wuji_version();
        if (!version || std::strcmp(version, "2026.8.31") != 0)
            throw std::runtime_error("需要精确的官方C SDK 2026.8.31，拒绝初始化其他版本");
        check(wuji_init(nullptr), "初始化官方SDK");
        auto connect_options = wuji_connect_options_default();
        connect_options.timeout_ms = options.timeout_ms;
        connect_options.retry_count = 0;
        connect_options.enable_bridge = false;
        connect_options.auto_time_sync_interval_enabled = false;
        const bool by_address = options.address && *options.address;
        WujiConnectTarget target{};
        target.kind = by_address ? WUJI_CONNECT_TARGET_KIND_ADDR : WUJI_CONNECT_TARGET_KIND_SN;
        target.value = by_address ? options.address : options.serial_number;
        static std::atomic<uint64_t> next_alias{0};
        const std::string alias = "real_hand_" + std::to_string(++next_alias);
        check(wuji_connect(&target, alias.c_str(), &connect_options, &device), "按显式身份连接");
        if (!device || !wuji_dev_is_connected(device)) throw std::runtime_error("SDK连接未完成");
        Identity identity;
        char serial[256]{};
        const int32_t length = wuji_dev_serial_number(device, serial, sizeof(serial));
        if (length <= 0 || length >= static_cast<int32_t>(sizeof(serial)) || std::strcmp(serial, options.serial_number))
            throw std::runtime_error("设备序列号缺失、截断或不匹配");
        identity.serial = serial;
        // 二代typed API在设备代际不匹配时返回Unsupported，不猜测dev_model十六进制编码。
        WujiHandedness side{};
        check(wuji_hand_2_get_handedness(device, &side), "核对二代手侧与型号");
        if (side != WUJI_HANDEDNESS_LEFT && side != WUJI_HANDEDNESS_RIGHT)
            throw std::runtime_error("二代手侧枚举未知");
        identity.hand = side == WUJI_HANDEDNESS_LEFT ? 0 : 1;
        if (identity.hand != options.hand) throw std::runtime_error("设备手侧与配置不符");
        uint8_t count = 0;
        check(wuji_hand_2_online_joints_count(device, &count), "读取二代在线关节数");
        if (count != 20) throw std::runtime_error("二代必须20个关节全部在线");
        identity.model = 2;
        // 固定C SDK不导出固件元数据；此值必须来自同设备的官方Python只读预检。
        // 这里不将approved值冒充实测，不猜测未公开的raw参数路径或编码。
        identity.firmware = options.preflight_firmware_version;
        // 与现有retarget输出及URDF限位的float32表示逐位一致，再提升为ABI double。
        constexpr float thumb[4][2] = {{-1.187f,1.291f},{-1.484f,0.698f},{-1.047f,1.570f},{-1.047f,1.570f}};
        constexpr float finger[4][2] = {{-1.047f,1.570f},{-0.698f,0.698f},{-1.047f,2.094f},{-1.047f,1.570f}};
        indices.fill(-1);
        for (uint8_t i = 0; i < 20; ++i) {
            const auto &limits = i < 4 ? thumb[i] : finger[i % 4];
            identity.lower[i] = limits[0]; identity.upper[i] = limits[1];
            char label[64]{};
            check(wuji_hand_2_joint_label(i, label, sizeof(label), nullptr), "核验关节语义");
            const std::string expected = std::string(fingers[i / 4]) + "_S" + std::to_string(i % 4 + 1);
            if (label != expected) throw std::runtime_error("SDK关节语义不符合五指各S1..S4顺序");
            // 固定SDK：反馈NID=总线*5+节点(1..4)，command joint_id为其减1。
            const int nid = wuji_hand_2_joint_id_from_bus_node(i / 4, i % 4 + 1) + 1;
            if (nid > 255 || indices[nid] != -1) throw std::runtime_error("SDK关节NID映射重复");
            indices[nid] = i;
        }
        check(wuji_hand_2_subscribe_joint_states(device, on_states, this, &states), "订阅二代位置");
        check(wuji_hand_2_subscribe_joint_diagnostics(device, on_diagnostics, this, &diagnostics), "订阅二代诊断");
        return identity;
    }
    Feedback feedback() override {
        std::lock_guard<std::mutex> lock(feedback_mutex);
        if (!stream_error.empty()) throw std::runtime_error(stream_error);
        if (closing.load()) throw std::runtime_error("SDK反馈回调失败或资源已关闭");
        return latest;
    }
    void prepare(const RealHandOptions &options) override {
        // 官方SDK没有这些数值默认；省略时保留设备配置，不写入项目自选增益/电流。
        if (options.effort_limit >= 0)
            check(wuji_hand_2_set_all_effort_limit(device, static_cast<float>(options.effort_limit)), "设置二代力限");
        if (options.mit_kp >= 0) {
            std::array<float, 20> kp{}, kd{};
            kp.fill(static_cast<float>(options.mit_kp)); kd.fill(static_cast<float>(options.mit_kd));
            check(wuji_hand_2_set_all_mit_params(device, kp.data(), kd.data()), "设置二代MIT增益");
        }
        check(wuji_hand_2_joint_command_publish(device, &publisher), "创建二代目标发布器");
    }
    void send(const Angles &angles) override {
        WujiJointCommand commands[20]{};
        for (size_t i = 0; i < 20; ++i) commands[i].position = static_cast<float>(angles[i]);
        check(wuji_joint_command_publisher_send(publisher, commands), "发送二代关节目标");
    }
    void enable() override {
        check(wuji_hand_2_enable(device, nullptr), "使能20关节");
    }
    void clear_fault() override {
        check(wuji_hand_2_clear_fault(device, nullptr), "清除20关节故障标志");
    }
    void stop(bool emergency) override {
        // 急停动作与电机失能是不同接口，不能依赖未约定的状态转移。
        // 即使急停请求失败，也尝试失能；保留各自错误，由核心核验新鲜诊断。
        std::string errors;
        if (emergency) {
            try { check(wuji_hand_2_emergency_stop(device), "请求整手急停"); }
            catch (const std::exception &e) { errors = e.what(); }
        }
        try { check(wuji_hand_2_disable(device, nullptr), "禁用20关节"); }
        catch (const std::exception &e) {
            if (!errors.empty()) errors += "; ";
            errors += e.what();
        }
        close_publisher();
        if (!errors.empty()) throw std::runtime_error(errors);
    }
    void release() override {
        closing.store(true);
        close_publisher();
        // 不持有feedback_mutex；sub_close会等待SDK回调线程退出。
        if (states) { wuji_sub_close(states); states = nullptr; }
        if (diagnostics) { wuji_sub_close(diagnostics); diagnostics = nullptr; }
        if (device) {
            WujiDevice *old = device; device = nullptr;
            const auto result = wuji_dev_disconnect(old);
            const char *detail = result == WUJI_STATUS_OK ? nullptr : wuji_last_error();
            const std::string error = detail ? detail : "SDK断开失败";
            wuji_dev_release(old);
            if (result != WUJI_STATUS_OK) throw std::runtime_error(error);
        }
    }
};
}
std::unique_ptr<Transport> make_transport() { return std::make_unique<SDKAdapter>(); }
}
