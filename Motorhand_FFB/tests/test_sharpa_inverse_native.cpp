// 真实MuJoCo首帧的微小超调曾使零电机限位残差触发SLSQP反复roundoff失败。
#include "common/inverse_retarget.hpp"
#include "common/session.hpp"
#include "hands/sharpa/model.hpp"
#include "simulation/fake_glove.hpp"
#include "ve33_model.hpp"
#include "encos_driver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace motorhand::native;

namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// 保留独立的受阻实测位置，模拟驱动提交的限速目标。
class BlockedHand final : public HandDevice<22> {
public:
    explicit BlockedHand(const sharpa::Model& model) {
        desc.identity = {"sharpa", "OFFLINE", "test", false};
        desc.joint_names = model.joint_names();
        desc.lower_rad = model.lower_limits();
        desc.upper_rad = model.upper_limits();
        desc.online_source = HandStatusSource::complete_state_packet;
        desc.enabled_source = HandStatusSource::device_report;
    }
    void connect() override {connected = true;}
    void enable() override {enabled = true;}
    void home(const std::atomic<bool>*) override {}
    void disable() override {enabled = false;}
    void close() override {connected = false;}
    const HandDescriptor<22>& descriptor() const override {return desc;}
    HandState<22> read() override {
        HandState<22> state;
        state.identity = desc.identity;
        state.online_source = desc.online_source;
        state.enabled_source = desc.enabled_source;
        state.connected = connected;
        state.online.fill(connected);
        state.enabled.fill(enabled);
        state.received_ns = now_ns();
        return state;
    }
    void send(const JointArray<22>& target, std::uint64_t deadline) override {
        submit_position(target, deadline);
    }
    JointArray<22> submit_position(const JointArray<22>& target, std::uint64_t deadline) override {
        if (now_ns() >= deadline) throw std::runtime_error("离线接管命令过期");
        for (std::size_t i = 0; i < target.size(); ++i)
            submitted[i] += std::clamp(target[i] - submitted[i], -.04, .04);
        return submitted;
    }
private:
    HandDescriptor<22> desc;
    JointArray<22> submitted{};
    bool connected{}, enabled{};
};

void verify_handover(const std::filesystem::path& root,
                     const std::array<EncosMotorCalibration, 12>& cal) {
    auto model = std::make_unique<sharpa::Model>(root/"Motorhand_FFB/assets/sharpa-hand", false);
    auto hand = std::make_unique<BlockedHand>(*model);
    auto glove = std::make_unique<simulation::FakeGlove>(cal);
    auto* manual = glove.get();
    ControlOptions options;
    options.asset_root = root/"assets/hand_motor_ve33";
    options.glove_mapping = root/"hardware/encos_dexhand/configs/sim/glove_mapping.json";
    options.glove_output = false;
    options.force_feedback = false;
    options.transition_ns = 1;
    options.takeover_motor_deadband_rad = .002;
    ControlSession<22> session(options, std::move(hand), std::move(model), nullptr, std::move(glove));
    session.connect();
    session.enable();

    Vec12 closed{};
    for (std::size_t i=4; i<closed.size(); ++i)
        closed[i] = std::clamp(.65, cal[i].model_min_rad, cal[i].model_max_rad);
    manual->set_manual_target(closed);
    std::this_thread::sleep_for(std::chrono::milliseconds(45));
    JointArray<22> requested{};
    requested[7] = .7;
    session.step_inference(requested, now_ns());
    auto before = session.step_inference(requested, now_ns());
    if (before.hand_target[7] != .04 || before.hand.position_rad[7] != 0.)
        throw std::runtime_error("回放必须区分限速后的提交值和受阻实测");
    session.request_takeover(true);
    auto taken = session.step_takeover();
    if (taken.glove.motor_rad != closed || taken.hand_target != before.hand_target)
        throw std::runtime_error("接管首帧必须保持上次提交目标");
    for (int i=0; i<100; ++i) {
        if (session.step_takeover().hand_target != before.hand_target)
            throw std::runtime_error("手套静止但接管目标仍在漂移");
    }
    auto noise = closed;
    noise[4] += .001;
    manual->set_manual_target(noise);
    std::this_thread::sleep_for(std::chrono::milliseconds(45));
    if (session.step_takeover().hand_target != before.hand_target)
        throw std::runtime_error("电机噪声消耗了接管保持偏移");

    manual->set_manual_target(Vec12{});
    std::this_thread::sleep_for(std::chrono::milliseconds(45));
    auto moved = session.step_takeover();
    if (moved.glove.motor_rad != Vec12{} || moved.hand_target == before.hand_target)
        throw std::runtime_error("主动张手后接管仍锁在原目标");
    session.request_takeover(false);
    if (session.step_inference(requested, now_ns()).hand_target != moved.hand_target)
        throw std::runtime_error("恢复回放首帧必须从实际接管目标接入");
    session.stop();
}
}

int main(int argc,char** argv){try{
    const auto root=argc>1?std::filesystem::path(argv[1]):std::filesystem::current_path();
    Ve33Model glove(root/"assets/hand_motor_ve33",false);
    sharpa::Model hand(root/"Motorhand_FFB/assets/sharpa-hand",false);
    // 只读本地标定元数据；不connect任何真实设备。
    EncosDriver metadata((root/"hardware/encos_dexhand/configs/sim/glove_mapping.json").string());
    const auto cal=metadata.calibration();std::array<std::array<double,2>,12> bounds{};
    for(int i=0;i<12;++i)bounds[i]={cal[i].model_min_rad,cal[i].model_max_rad};
    const auto zero=glove.zero();
    const JointArray<22> measured={3.0788083443515223e-09,-7.580783990157879e-06,-9.400578659612215e-07,-5.394254973173509e-07,-3.1742723486461034e-07,3.064992414154025e-09,2.240168583138159e-08,-2.1177682149355153e-07,-1.1138036691935847e-07,3.064992414154163e-09,2.240168583138132e-08,-2.1177682149355222e-07,-1.1138036691935875e-07,3.0649924141540377e-09,2.2401685831381125e-08,-2.1177682149355222e-07,-1.1138036691935875e-07,-7.929232198664313e-11,3.0815223720880726e-09,2.2444141595400738e-08,-2.117726690279212e-07,-1.1137831519296002e-07};
    for(const auto& q:{JointArray<22>{},measured,hand.solve(zero.points_m)}) {
        CommonInverseRetarget<22> inverse(glove,hand,bounds);
        const auto result=inverse.solve(q,zero);
        if(!result.success)throw std::runtime_error(result.failure_reason);
        if(!std::isfinite(result.task_error_rms_mm))throw std::runtime_error("逆解误差非有限");
        for(int i=0;i<12;++i)if(result.pose.motor_rad[i]<bounds[i][0]||result.pose.motor_rad[i]>bounds[i][1])
            throw std::runtime_error("最终电机目标必须严格满足原始标定限位");
    }
    verify_handover(root, cal);
    std::cout<<"Sharpa共同逆解、限速锚点与接管往返保持通过\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
