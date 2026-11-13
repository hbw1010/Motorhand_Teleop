#include "fake_sdk.h"
#include <wuji_sdk.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <thread>

namespace {
std::mutex settings_mutex;
std::deque<std::shared_ptr<fake_sdk::Scenario>> pending;
std::string sdk_version = "2026.8.31";
std::atomic<int> initializations{0}, default_hand{1};
std::array<float, 20> default_velocities{}, default_effort{};
constexpr const char *fingers[] = {"thumb", "index", "middle", "ring", "pinky"};
}
struct WujiDevice { std::shared_ptr<fake_sdk::Scenario> scenario; bool connected = true; };
struct WujiJointCommandPublisher { std::shared_ptr<fake_sdk::Scenario> scenario; };
struct WujiSub {
    std::shared_ptr<fake_sdk::Scenario> scenario;
    std::atomic<bool> stop{false};
    std::thread worker;
};
namespace fake_sdk {
void enqueue(const std::shared_ptr<Scenario> &scenario) {
    std::lock_guard<std::mutex> lock(settings_mutex); pending.push_back(scenario);
}
void set_version(const char *version) {
    std::lock_guard<std::mutex> lock(settings_mutex); sdk_version = version;
}
int initialization_count() { return initializations.load(); }
}
namespace {
template<class Callback> void stream(WujiSub *sub, Callback callback) {
    uint64_t sequence = 0;
    bool acknowledged_pause = false;
    while (!sub->stop.load()) {
        const bool paused = sub->scenario->pause.load();
        if (paused != acknowledged_pause) {
            sub->scenario->paused_workers.fetch_add(paused ? 1 : -1);
            acknowledged_pause = paused;
        }
        if (!paused) callback(++sequence);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (acknowledged_pause) sub->scenario->paused_workers.fetch_sub(1);
}
}
extern "C" {
void real_hand_test_set_default_hand(int hand) { default_hand.store(hand); }
void real_hand_test_set_default_feedback(const float *velocities_rad_s, const float *effort_a) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    std::copy_n(velocities_rad_s, 20, default_velocities.begin());
    std::copy_n(effort_a, 20, default_effort.begin());
}
const char *wuji_version(void) { return sdk_version.c_str(); }
const char *wuji_last_error(void) { return "离线C SDK注入错误"; }
WujiStatus wuji_init(const WujiInitOptions *) { ++initializations; return WUJI_STATUS_OK; }
WujiConnectOptions wuji_connect_options_default(void) { return {}; }
WujiStatus wuji_connect(const WujiConnectTarget *target, const char *, const WujiConnectOptions *, WujiDevice **out) {
    std::shared_ptr<fake_sdk::Scenario> scenario;
    {
        std::lock_guard<std::mutex> lock(settings_mutex);
        if (!pending.empty()) { scenario = pending.front(); pending.pop_front(); }
        else {
            scenario = std::make_shared<fake_sdk::Scenario>();
            scenario->hand = default_hand.load();
            scenario->velocities_rad_s = default_velocities;
            scenario->effort_a = default_effort;
        }
    }
    if (scenario->serial.empty()) scenario->serial = target->value;
    *out = new WujiDevice{std::move(scenario), true};
    return WUJI_STATUS_OK;
}
uint8_t wuji_dev_is_connected(WujiDevice *device) { return device && device->connected; }
int32_t wuji_dev_serial_number(WujiDevice *device, char *buffer, size_t capacity) {
    if (capacity) std::snprintf(buffer, capacity, "%s", device->scenario->serial.c_str());
    return static_cast<int32_t>(device->scenario->serial.size());
}
WujiStatus wuji_hand_2_get_handedness(WujiDevice *device, WujiHandedness *side) {
    if (!device->scenario->supported) return WUJI_STATUS_ERR_UNSUPPORTED;
    *side = device->scenario->hand == 0 ? WUJI_HANDEDNESS_LEFT : WUJI_HANDEDNESS_RIGHT;
    return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_online_joints_count(WujiDevice *device, uint8_t *count) {
    *count = device->scenario->online_count; return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_joint_label(uint8_t index, char *buffer, size_t capacity, size_t *needed) {
    if (index >= 20) return WUJI_STATUS_ERR_INVALID_ARG;
    char label[32]{};
    std::snprintf(label, sizeof(label), "%s_S%d", fingers[index / 4], index % 4 + 1);
    if (needed) *needed = std::strlen(label) + 1;
    if (capacity) std::snprintf(buffer, capacity, "%s", label);
    return capacity > std::strlen(label) ? WUJI_STATUS_OK : WUJI_STATUS_ERR_BUFFER_TOO_SMALL;
}
uint8_t wuji_hand_2_joint_id_from_bus_node(uint8_t bus, uint8_t node) { return bus * 5 + node - 1; }
WujiStatus wuji_hand_2_decode_joint_status(uint32_t word, WujiJointStatus *status) {
    *status = {};
    status->ext_state = word & 255;
    status->enabled = (word & 256) != 0;
    return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_describe_error(uint16_t code, WujiErrorInfo *out) {
    if (!out) return WUJI_STATUS_ERR_INVALID_ARG;
    *out = {};
    if (code == fake_sdk::synthetic_lookup_failure) return WUJI_STATUS_ERR_INTERNAL;
    const char *name = nullptr, *severity = nullptr;
    switch (code) {
        case fake_sdk::warning: name = "Enc1BitRate"; severity = "Warning"; break;
        case fake_sdk::deferred_stop: name = "TestDeferredAuto"; severity = "DeferredStop"; break;
        case fake_sdk::immediate_stop: name = "TestImmediateAuto"; severity = "ImmediateStop"; break;
        case fake_sdk::synthetic_fatal: name = "SyntheticFatal"; severity = "Fatal"; break;
        case fake_sdk::synthetic_unknown_severity: name = "SyntheticSeverity"; severity = "WarningExtra"; break;
        case fake_sdk::synthetic_warning: name = "SyntheticWarning"; severity = "Warning"; break;
        default: return WUJI_STATUS_ERR_NOT_FOUND;
    }
    out->code = code;
    out->is_test = code != fake_sdk::warning;
    std::snprintf(out->name, sizeof(out->name), "%s", name);
    std::snprintf(out->severity, sizeof(out->severity), "%s", severity);
    std::snprintf(out->clear_policy, sizeof(out->clear_policy), "%s", "AutoClear");
    return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_subscribe_joint_states(WujiDevice *device, WujiJointStateFrameSubCallback callback,
                                           void *data, WujiSub **out) {
    auto *sub = new WujiSub;
    sub->scenario = device->scenario;
    sub->worker = std::thread([sub, callback, data] {
        stream(sub, [&](uint64_t sequence) {
            const int fault = sub->scenario->state_fault.load();
            std::array<float, 20> positions, velocities, effort;
            {
                std::lock_guard<std::mutex> lock(sub->scenario->mutex);
                positions = sub->scenario->positions;
                velocities = sub->scenario->velocities_rad_s;
                effort = sub->scenario->effort_a;
            }
            WujiJointStateEntry entries[20]{};
            // 故意逆序提供，消费者必须按真实NID重排，不得依赖数组顺序。
            for (int i = 0; i < 20; ++i) {
                const int index = 19 - i;
                entries[i].nid = (index / 4) * 5 + index % 4 + 1;
                entries[i].position = positions[index];
                entries[i].velocity = velocities[index];
                entries[i].effort = effort[index];
            }
            if (fault == 2) entries[19].nid = entries[0].nid;
            if (fault == 3) entries[19].nid = 255;
            if (fault == 4) entries[19].position = std::numeric_limits<float>::quiet_NaN();
            WujiJointStateFrame frame{};
            frame.header.seq = sequence; frame.header.timestamp_us = sequence * 2000;
            frame.num_joints = fault == 1 ? 19 : 20;
            frame.joints = entries; frame.joints_len = frame.num_joints;
            callback(WUJI_FRAME_KIND_OK, &frame, data);
        });
        callback(WUJI_FRAME_KIND_END, nullptr, data);
    });
    *out = sub; return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_subscribe_joint_diagnostics(WujiDevice *device, WujiJointDiagnosticsFrameSubCallback callback,
                                                void *data, WujiSub **out) {
    auto *sub = new WujiSub;
    sub->scenario = device->scenario;
    sub->worker = std::thread([sub, callback, data] {
        stream(sub, [&](uint64_t sequence) {
            const int fault = sub->scenario->diagnostic_fault.load();
            if (fault == 7) { callback(WUJI_FRAME_KIND_ERROR, nullptr, data); return; }
            WujiJointDiagnosticsEntry entries[20]{};
            for (int i = 0; i < 20; ++i) {
                const int index = 19 - i;
                entries[i].nid = (index / 4) * 5 + index % 4 + 1;
                entries[i].status_word = sub->scenario->enabled.load() ? 258 : 1;
            }
            if (fault == 2) entries[19].nid = entries[0].nid;
            if (fault == 3) entries[19].nid = 255;
            if (fault == 4) entries[19].status_word = 4;
            if (fault == 5) entries[19].status_word = 2;
            if (fault == 6) entries[19].error_code_current = 0x3001;
            entries[11].error_code_current = sub->scenario->diagnostic_code.load();
            if (fault == 8) entries[11].error_code_current = fake_sdk::immediate_stop;
            WujiJointDiagnosticsFrame frame{};
            frame.header.seq = sequence; frame.header.timestamp_us = sequence * 2000;
            frame.num_joints = fault == 1 ? 19 : 20;
            frame.joints = entries; frame.joints_len = frame.num_joints;
            callback(WUJI_FRAME_KIND_OK, &frame, data);
        });
        callback(WUJI_FRAME_KIND_END, nullptr, data);
    });
    *out = sub; return WUJI_STATUS_OK;
}
void wuji_sub_close(WujiSub *sub) {
    if (!sub) return;
    sub->stop.store(true);
    sub->worker.join();
    ++sub->scenario->closed_subs;
    delete sub;
}
WujiStatus wuji_hand_2_set_all_effort_limit(WujiDevice *device, float) {
    ++device->scenario->effort_writes; return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_set_all_mit_params(WujiDevice *device, const float *, const float *) {
    ++device->scenario->mit_writes; return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_joint_command_publish(WujiDevice *device, WujiJointCommandPublisher **out) {
    *out = new WujiJointCommandPublisher{device->scenario}; return WUJI_STATUS_OK;
}
WujiStatus wuji_joint_command_publisher_send(WujiJointCommandPublisher *publisher, const WujiJointCommand *commands) {
    if (publisher->scenario->send_failure.load()) return WUJI_STATUS_ERR_DISCONNECTED;
    std::lock_guard<std::mutex> lock(publisher->scenario->mutex);
    for (int i = 0; i < 20; ++i) publisher->scenario->positions[i] = commands[i].position;
    ++publisher->scenario->sends;
    return WUJI_STATUS_OK;
}
void wuji_joint_command_publisher_close(WujiJointCommandPublisher *publisher) { delete publisher; }
WujiStatus wuji_hand_2_enable(WujiDevice *device, const uint8_t *) {
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (static_cast<uint64_t>(now) < device->scenario->clear_ready_ns.load()) {
        ++device->scenario->ignored_enables;
        return WUJI_STATUS_OK;
    }
    device->scenario->enabled.store(true);
    if (device->scenario->enable_fault) device->scenario->diagnostic_fault = 8;
    return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_disable(WujiDevice *device, const uint8_t *) {
    ++device->scenario->stops;
    if (device->scenario->disable_failure) return WUJI_STATUS_ERR_DISCONNECTED;
    device->scenario->enabled.store(false); return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_emergency_stop(WujiDevice *device) {
    ++device->scenario->emergencies;
    if (device->scenario->emergency_failure) return WUJI_STATUS_ERR_DISCONNECTED;
    if (!device->scenario->emergency_keeps_enabled) device->scenario->enabled.store(false);
    return WUJI_STATUS_OK;
}
WujiStatus wuji_hand_2_clear_fault(WujiDevice *device, const uint8_t *) {
    ++device->scenario->clears;
    if (device->scenario->clear_failure) return WUJI_STATUS_ERR_DISCONNECTED;
    if (device->scenario->enabled) return WUJI_STATUS_ERR_INVALID_ARG;
    device->scenario->clear_ready_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() +
        static_cast<uint64_t>(device->scenario->clear_busy_ms.load()) * 1000000;
    if (!device->scenario->sticky_fault) {
        device->scenario->diagnostic_fault = 0;
        device->scenario->diagnostic_code = 0;
    }
    return WUJI_STATUS_OK;
}
WujiStatus wuji_dev_disconnect(WujiDevice *device) { device->connected = false; return WUJI_STATUS_OK; }
void wuji_dev_release(WujiDevice *device) { ++device->scenario->releases; delete device; }
}
