#include "encos_driver.hpp"
#include "playback_control.hpp"
#include "session.hpp"
#include "telemetry.hpp"
#include "trajectory.hpp"
#include "wuji_device.hpp"
#include "wuji_identity.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace motorhand::native;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t timeout_ns = 100'000'000;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}
void require_fresh(std::uint64_t stamp, std::uint64_t now) {
    if (!stamp || stamp > now || now - stamp >= timeout_ns)
        throw std::runtime_error("Wuji反馈已过期或不属于单调时钟域");
}
struct Args {
    std::string command;
    std::filesystem::path trajectory;
    std::string serial;
    std::string firmware = "2.2.3";
    bool left = false;
    bool execute = false;
    int rate = 50;
    int curves_port = 0;
    double duration_s = 0;
};
Args parse(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("用法: motorhand-native {teleop|force-contact|force-linear|record|replay|takeover} [轨迹] [--execute] [--curves-port 端口]");
    Args out; out.command = argv[1];
    if (out.command != "teleop" && out.command != "force-contact" && out.command != "force-linear" && out.command != "record" && out.command != "replay" && out.command != "takeover")
        throw std::invalid_argument("未知控制命令");
    for (int i = 2; i < argc; ++i) {
        const std::string token = argv[i];
        auto value = [&]() -> std::string { if (++i >= argc) throw std::invalid_argument("选项缺少参数: " + token); return argv[i]; };
        if (token == "--execute") out.execute = true;
        else if (token == "--wuji-serial") out.serial = value();
        else if (token == "--approved-firmware") out.firmware = value();
        else if (token == "--hand") { auto v = value(); if (v != "left" && v != "right") throw std::invalid_argument("hand必须为left或right"); out.left = v == "left"; }
        else if (token == "--rate") out.rate = std::stoi(value());
        else if (token == "--curves-port") out.curves_port = std::stoi(value());
        else if (token == "--duration") out.duration_s = std::stod(value());
        else if (!token.empty() && token[0] != '-' && out.trajectory.empty()) out.trajectory = token;
        else throw std::invalid_argument("不支持的参数: " + token);
    }
    if (out.rate < 1 || out.rate > 200 || out.curves_port < 0 || out.curves_port > 65535 ||
        !std::isfinite(out.duration_s) || out.duration_s < 0 || out.firmware.empty())
        throw std::invalid_argument("频率、时长、固件版本或曲线端口非法");
    if ((out.command == "record" || out.command == "replay" || out.command == "takeover") && out.trajectory.empty())
        throw std::invalid_argument("必须指定轨迹文件路径");
    return out;
}
SessionOptions session_options(const Args& args) {
    const auto root = std::filesystem::current_path();
    SessionOptions options;
    options.asset_root = root / "assets/hand_motor_ve33";
    options.retarget_yaml = root / "config/retargeting/official" / (args.left ? "left.yaml" : "right.yaml");
    options.glove_mapping = root / "hardware/encos_dexhand/configs/glove_mapping.json";
    options.linear_feedback = root / "config/feedback/max-torque-linear-right.json";
    options.contact_feedback = root / "config/feedback/contact-impedance-right.json";
    options.wuji_serial = args.serial;
    options.approved_firmware = args.firmware;
    options.left = args.left;
    options.force_feedback = args.command == "takeover" || args.command == "force-contact" || args.command == "force-linear";
    options.linear_force_feedback = args.command == "force-linear";
    options.glove_output = options.force_feedback;
    return options;
}
void print_torque(const SessionTick& tick, const std::array<EncosMotorCalibration, 12>& calibration) {
    std::array<double, 12> raw{};
    for (int model = 0; model < 12; ++model) {
        const auto& axis = calibration[model];
        raw.at(static_cast<std::size_t>(axis.raw_motor_id - 1)) = tick.glove_target[model] * axis.sign * axis.scale;
    }
    std::cout << "手套MIT前馈[名义Nm,串口已提交,";
    if (tick.linear_force) std::cout << "线性电流估计";
    else switch (tick.contact_phase) {
        case ContactPhase::waiting_geometry: std::cout << "等待几何"; break;
        case ContactPhase::contact: std::cout << "接触估计"; break;
        case ContactPhase::arming: std::cout << "判据确认中"; break;
        case ContactPhase::releasing: std::cout << "释放中"; break;
        case ContactPhase::free: std::cout << "自由"; break;
        default: std::cout << "未配置";
    }
    std::cout << "]" << std::fixed << std::setprecision(3);
    for (int i = 0; i < 12; ++i) std::cout << " J" << i+1 << '=' << std::showpos << raw[i] << std::noshowpos;
    std::cout << std::endl;
}
void record_tick(TrajectoryRecorder& recorder, const SessionTick& tick) {
    TrajectoryFrame frame;
    frame.source_ns = tick.hand.received_ns;
    frame.position_rad = tick.hand.position_rad;
    frame.velocity_rad_s = tick.hand.velocity_rad_s;
    frame.effort_a = tick.hand.effort_a;
    frame.target_rad = tick.wuji_target;
    frame.has_velocity = tick.hand.has_velocity;
    frame.has_effort = tick.hand.has_effort;
    recorder.record(frame);
}
void sleep_cycle(Clock::time_point start, int rate) {
    std::this_thread::sleep_until(start + std::chrono::nanoseconds(1'000'000'000 / rate));
}
int live(const Args& args) {
    auto options = session_options(args);
    if (args.left && options.force_feedback) throw std::runtime_error("当前力反馈配置仅对右手校准");
    Session session(options);
    std::unique_ptr<TelemetryPublisher> telemetry;
    if (args.curves_port) telemetry = std::make_unique<TelemetryPublisher>(args.curves_port, options.glove_mapping);
    std::array<EncosMotorCalibration, 12> calibration{};
    if (options.force_feedback) {
        EncosDriver calibrated(options.glove_mapping.string(), options.linear_feedback.string());
        calibration = calibrated.calibration();
    }
    std::cout << "正在只读预检设备；MotorHand只读轮询包含0A探针。" << std::endl;
    const auto identity = session.connect();
    std::unique_ptr<TrajectoryRecorder> recorder;
    if (args.command == "record") recorder = std::make_unique<TrajectoryRecorder>(args.trajectory, identity.left ? "left" : "right", identity.serial, identity.firmware);
    try {
        std::cout << (options.glove_output ? "即将回零手套并使能两侧。" : "即将使能Wuji；手套仅作0A只读轮询。") << std::endl;
        session.enable(); session.request_takeover(true);
        const auto start = Clock::now(); auto report = start;
        while (!interrupted) {
            const auto cycle = Clock::now();
            if (args.duration_s > 0 && std::chrono::duration<double>(cycle-start).count() >= args.duration_s) break;
            auto tick = session.step_takeover();
            if (recorder) record_tick(*recorder, tick);
            if (telemetry) telemetry->publish(tick);
            if (options.force_feedback && cycle >= report) {
                if (tick.force_commanded && tick.glove_submit_ns) print_torque(tick, calibration);
                else std::cout << "手套前馈：等待异步串口提交结果。" << std::endl;
                report = cycle + std::chrono::seconds(1);
            }
            sleep_cycle(cycle, args.rate);
        }
        if (recorder) recorder->finish();
        session.stop();
        return interrupted ? 130 : 0;
    } catch (...) {
        if (recorder) { try { recorder->abort(); } catch (...) {} }
        throw;
    }
}
int replay_hand_only(const Args& args, const Trajectory& trajectory) {
    auto identity = inspect_wuji(args.serial.empty() ? trajectory.serial() : args.serial, args.left);
    if (identity.serial != trajectory.serial() || identity.firmware != trajectory.firmware() ||
        (identity.left ? "left" : "right") != trajectory.hand() || identity.firmware != args.firmware)
        throw std::runtime_error("轨迹身份或固件不匹配");
    WujiDevice hand(identity, args.firmware); hand.connect();
    const auto state = hand.read();
    Vec20 lower{}, upper{}, anchor{};
    for (int i = 0; i < 20; ++i) { lower[i] = state.lower_limits[i]; upper[i] = state.upper_limits[i]; }
    trajectory.validate_limits(lower, upper);
    try {
        hand.enable();
        const auto enabled_state = hand.read();
        require_fresh(enabled_state.timestamp_ns, now_ns());
        for (int i = 0; i < 20; ++i) {
            if (!enabled_state.online[i] || !enabled_state.enabled[i])
                throw std::runtime_error("Wuji 使能后反馈不完整");
            anchor[i] = enabled_state.positions[i];
        }
        const auto start = now_ns();
        PlaybackClock clock(trajectory.duration_ns(), start);
        while (!interrupted) {
            const auto cycle = Clock::now(); const auto now = now_ns();
            const auto position = clock.position(now);
            auto target = trajectory.at(position);
            const double x = std::min(1.0, (now - start) / 500'000'000.0);
            const double a = x*x*x*(10 + x*(-15 + 6*x));
            for (int i = 0; i < 20; ++i) target[i] = anchor[i] + a*(target[i]-anchor[i]);
            const auto current = hand.read(); require_fresh(current.timestamp_ns, now_ns());
            hand.send(target, current.timestamp_ns + timeout_ns);
            if (position >= trajectory.duration_ns()) break;
            sleep_cycle(cycle, args.rate);
        }
        hand.disable(); hand.close();
        return interrupted ? 130 : 0;
    } catch (const std::exception& original) {
        std::string errors = original.what();
        try { hand.disable(); } catch (const std::exception& error) { errors += std::string("; 失能失败: ") + error.what(); }
        try { hand.close(); } catch (const std::exception& error) { errors += std::string("; 断开失败: ") + error.what(); }
        throw std::runtime_error(errors);
    }
}
int replay(const Args& args) {
    Trajectory trajectory(args.trajectory);
    if (trajectory.hand() != (args.left ? "left" : "right") || (!args.serial.empty() && args.serial != trajectory.serial()) || trajectory.firmware() != args.firmware)
        throw std::runtime_error("轨迹手侧、序列号或批准固件不符");
    std::cout << "轨迹已校验：" << std::fixed << std::setprecision(3) << trajectory.duration_ns()/1e9 << "秒；正常倍速。" << std::endl;
    if (!args.execute) { std::cout << "仅验证文件；未连接、使能或控制设备。" << std::endl; return 0; }
    if (args.command == "replay") return replay_hand_only(args, trajectory);
    TerminalKeys keys;
    if (!keys.active()) throw std::runtime_error("手套接管需要交互终端");
    auto options = session_options(args);
    if (args.left) throw std::runtime_error("当前力反馈配置仅对右手校准");
    options.wuji_serial = trajectory.serial();
    Session session(options);
    std::unique_ptr<TelemetryPublisher> telemetry;
    if (args.curves_port) telemetry = std::make_unique<TelemetryPublisher>(args.curves_port, options.glove_mapping);
    EncosDriver calibrated(options.glove_mapping.string(), options.linear_feedback.string());
    const auto calibration = calibrated.calibration();
    session.connect(); session.validate_trajectory(trajectory);
    std::cout << "即将伺服回零手套并使能两侧；按 T/空格切换接管与恢复，Q退出。" << std::endl;
    session.enable();
    PlaybackClock clock(trajectory.duration_ns(), now_ns(), options.transition_ns);
    bool inverse_ready = false;
    auto report = Clock::now();
    while (!interrupted) {
        const auto cycle = Clock::now();
        for (auto event = keys.read(); event != KeyEvent::none; event = keys.read()) {
            if (event == KeyEvent::quit) { session.stop(); return 0; }
            if (event == KeyEvent::toggle) {
                const auto now = now_ns();
                if (clock.playing()) { clock.pause(now, inverse_ready); session.request_takeover(true); std::cout << "手套接管；回放暂停。" << std::endl; }
                else { session.request_takeover(false); clock.resume(now); inverse_ready = false; std::cout << "恢复回放。" << std::endl; }
            }
        }
        const auto stamp = now_ns();
        const auto position = clock.position(stamp, inverse_ready);
        SessionTick tick = clock.playing() ? session.step_inference(trajectory.at(position), stamp) : session.step_takeover();
        inverse_ready = tick.inverse_fresh;
        if (telemetry) telemetry->publish(tick);
        if (cycle >= report) {
            if (tick.force_commanded && tick.takeover) print_torque(tick, calibration);
            else std::cout << "回放 " << std::fixed << std::setprecision(2) << position/1e9 << '/' << trajectory.duration_ns()/1e9 << "s" << (inverse_ready ? "" : "（等待新鲜手套逆解）") << std::endl;
            report = cycle + std::chrono::seconds(1);
        }
        if (clock.playing() && inverse_ready && position >= trajectory.duration_ns()) break;
        sleep_cycle(cycle, args.rate);
    }
    session.stop();
    return interrupted ? 130 : 0;
}
} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal);
    try {
        const auto args = parse(argc, argv);
        if (args.command == "replay" || args.command == "takeover") return replay(args);
        if (!args.execute) throw std::invalid_argument("实时控制必须显式提供 --execute");
        return live(args);
    } catch (const std::exception& error) {
        std::cerr << "控制失败，停止本会话：" << error.what() << std::endl;
        return 1;
    }
}
