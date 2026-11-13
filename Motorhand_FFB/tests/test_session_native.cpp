// Offline-only Session integration. Link this file INSTEAD OF wuji_device.cpp,
// wuji_identity.cpp and encos_driver.cpp; see the main-agent handoff for the
// production sources and libraries. No physical device API is linked here.
#include "motorhand_teleop/real_hand/native_control/session.hpp"
#include "motorhand_teleop/real_hand/native_control/trajectory.hpp"
#include "motorhand_teleop/real_hand/native_control/wuji_device.hpp"
#include "motorhand_teleop/real_hand/native_control/encos_driver.hpp"
#include "motorhand_teleop/real_hand/native_control/ve33_model.hpp"
#include "motorhand_teleop/real_hand/native_control/wuji_retarget.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace motorhand::native;
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr const char* serial = "OFFLINE-SESSION-ONLY";
constexpr const char* firmware = "offline-approved";
std::uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void must_throw(F&& f, const char* why) {
    bool caught = false; try { f(); } catch (const std::exception&) { caught = true; }
    check(caught, why);
}
double max_abs(const Vec12& v) { double x=0; for (double a : v) x=std::max(x,std::abs(a)); return x; }
double distance(const Vec20& a, const Vec20& b) { double x=0; for (int i=0; i<20; ++i) x=std::max(x,std::abs(a[i]-b[i])); return x; }

// Only this boundary is fake. A dedicated producer generates receive frames on
// the real steady clock every 20 ms; getters return the cached timestamps.
struct Sim {
    struct GloveWrite { bool force; Vec12 value; std::uint64_t submitted, deadline; };
    struct HandWrite { Vec20 value; std::uint64_t submitted, deadline; };
    std::mutex mutex;
    std::atomic<bool> running{false};
    std::thread producer;
    Vec20 hand_position{}, hand_target{};
    Vec12 motors{}, desired_motors{}, manual_motors{};
    RealHandState hand_frame{};
    GloveFrame glove_frame{};
    bool hand_enabled=false, glove_connected=false, hand_connected=false, glove_homed=false;
    bool force_enabled=false, manual=false, frozen=false, loaded=false, stale_hand=false;
    bool fail_hand_send=false, fail_glove_send=false;
    int hand_disable=0, glove_disable=0, hand_close=0, glove_close=0, expired_attempts=0;
    int hand_homes=0; bool fail_hand_home=false;
    int loaded_joint=-1; double hand_limit=3.0;
    std::vector<HandWrite> hand_writes;
    std::vector<GloveWrite> glove_writes;
    void reset(const Vec20& initial) {
        std::lock_guard<std::mutex> lock(mutex);
        hand_position=initial; hand_target=initial; motors={}; desired_motors={}; manual_motors={};
        hand_frame={}; glove_frame={}; hand_enabled=glove_connected=hand_connected=glove_homed=false;
        force_enabled=manual=frozen=loaded=stale_hand=fail_hand_send=fail_glove_send=false;
        hand_disable=glove_disable=hand_close=glove_close=expired_attempts=0;
        hand_homes=0; fail_hand_home=false;
        loaded_joint=-1; hand_limit=3.0;
        hand_writes.clear(); glove_writes.clear();
    }
    void start() {
        running.store(true);
        producer = std::thread([this] {
            auto next=Clock::now();
            while (running.load()) {
                next += std::chrono::milliseconds(20);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto stamp=now_ns();
                    motors = manual ? manual_motors : desired_motors;
                    for (int i=0; i<12; ++i) {
                        glove_frame.motor_rad[i]=motors[i];
                        glove_frame.raw_degrees[i]=motors[i]*180/pi;
                        glove_frame.measured_current_a[i]=0;
                        glove_frame.received_ns[i]=stamp;
                    }
                    glove_frame.oldest_ns=stamp;
                    for (int i=0; i<20; ++i) {
                        hand_frame.positions[i]=hand_position[i];
                        hand_frame.velocities_rad_s[i]=0;
                        hand_frame.effort_a[i]=loaded && (loaded_joint<0 || loaded_joint==i) ? 1.0 : 0.0;
                        hand_frame.lower_limits[i]=-hand_limit; hand_frame.upper_limits[i]=hand_limit;
                        hand_frame.online[i]=1; hand_frame.enabled[i]=hand_enabled ? 1 : 0;
                    }
                    hand_frame.has_velocity=hand_frame.has_effort=1;
                    hand_frame.phase=hand_enabled ? 2 : 1;
                    if (!stale_hand) hand_frame.timestamp_ns=stamp;
                }
                std::this_thread::sleep_until(next);
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    void stop() { running.store(false); if (producer.joinable()) producer.join(); }
    ~Sim() { stop(); }
} sim;

namespace motorhand::native {
WujiIdentity inspect_wuji(const std::string& requested, bool left, std::uint32_t) {
    if (left || (!requested.empty() && requested != serial)) throw std::runtime_error("offline identity mismatch");
    return {serial, "offline-only", firmware, false};
}
WujiDevice::WujiDevice(const WujiIdentity& identity, const std::string& approved) {
    if (identity.serial != serial || approved != firmware) throw std::runtime_error("offline Wuji identity mismatch");
}
WujiDevice::~WujiDevice() noexcept = default;
void WujiDevice::connect() { std::lock_guard<std::mutex> l(sim.mutex); sim.hand_connected=true; }
RealHandState WujiDevice::read() { std::lock_guard<std::mutex> l(sim.mutex); return sim.hand_frame; }
void WujiDevice::enable() {
    { std::lock_guard<std::mutex> l(sim.mutex); sim.hand_enabled=true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
}
void WujiDevice::home(const std::atomic<bool>* stop_requested) {
    if (stop_requested && stop_requested->load()) throw std::runtime_error("cancelled home");
    { std::lock_guard<std::mutex> l(sim.mutex);
      check(sim.glove_homed && sim.hand_enabled,"Wuji回零必须在手套回零和Wuji使能后");
      ++sim.hand_homes;
      if(sim.fail_hand_home)throw std::runtime_error("injected Wuji home failure");
      sim.hand_target.fill(0);sim.hand_position.fill(0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
}
void WujiDevice::send(const Vec20& q, std::uint64_t deadline) {
    std::lock_guard<std::mutex> l(sim.mutex);
    if (sim.fail_hand_send) { sim.fail_hand_send=false; throw std::runtime_error("injected Wuji send failure"); }
    const auto stamp=now_ns();
    if (stamp>=deadline) { ++sim.expired_attempts; throw std::runtime_error("expired Wuji send"); }
    sim.hand_writes.push_back({q,stamp,deadline}); sim.hand_target=q;
    if (!sim.frozen) sim.hand_position=q;
}
void WujiDevice::disable() { std::lock_guard<std::mutex> l(sim.mutex); ++sim.hand_disable; sim.hand_enabled=false; }
void WujiDevice::close() { std::lock_guard<std::mutex> l(sim.mutex); ++sim.hand_close; sim.hand_connected=false; }

struct EncosDriver::Impl {};
EncosDriver::EncosDriver(const std::string&, const std::string&) : impl_(std::make_unique<Impl>()) {}
EncosDriver::~EncosDriver() = default;
void EncosDriver::connect() { std::lock_guard<std::mutex> l(sim.mutex); sim.glove_connected=true; }
GloveFrame EncosDriver::read() { std::lock_guard<std::mutex> l(sim.mutex); return sim.glove_frame; }
std::array<EncosMotorCalibration,12> EncosDriver::calibration() const {
    std::array<EncosMotorCalibration,12> result{};
    for (int i=0; i<12; ++i) result[i]={i+1,0,1,1,-180,180,-1.0,1.0,10.0,true};
    return result;
}
std::array<std::uint64_t,12> EncosDriver::last_command_write_ns() const {
    std::lock_guard<std::mutex> l(sim.mutex); std::array<std::uint64_t,12> v{};
    if (!sim.glove_writes.empty()) v.fill(sim.glove_writes.back().submitted);
    return v;
}
void EncosDriver::home(double, double) {
    std::lock_guard<std::mutex> l(sim.mutex); sim.glove_homed=true; sim.desired_motors={};
}
void EncosDriver::enable_force() {
    std::lock_guard<std::mutex> l(sim.mutex);
    if (!sim.glove_homed) throw std::runtime_error("offline force before home");
    sim.force_enabled=true;
}
static Vec12 submit_glove(Vec12 value, std::uint64_t deadline, bool force_mode) {
    // Models one serial frame completing after the real 12-axis asynchronous
    // worker; it never fabricates a submission timestamp at enqueue.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::lock_guard<std::mutex> l(sim.mutex);
    if (sim.fail_glove_send) { sim.fail_glove_send=false; throw std::runtime_error("injected Encos send failure"); }
    const auto stamp=now_ns();
    if (stamp>=deadline) { ++sim.expired_attempts; throw std::runtime_error("expired Encos send"); }
    if (force_mode) {
        if (!sim.force_enabled) throw std::runtime_error("offline force not enabled");
        for (double& v : value) {
            const auto raw=std::lround((std::clamp(v,-10.0,10.0)+10.0)*4095.0/20.0);
            v=-10.0+raw*20.0/4095.0;
        }
    } else { sim.force_enabled=false; sim.desired_motors=value; }
    sim.glove_writes.push_back({force_mode,value,stamp,deadline});
    return value;
}
Vec12 EncosDriver::write_position(const Vec12& value, std::uint64_t deadline) { return submit_glove(value,deadline,false); }
Vec12 EncosDriver::write_torque(const Vec12& value, std::uint64_t deadline) { return submit_glove(value,deadline,true); }
DriverImpedanceSubmission EncosDriver::write_impedance(const DriverImpedanceCommand& c,std::uint64_t deadline) {
    DriverImpedanceSubmission r;r.command=c;
    {std::lock_guard<std::mutex> l(sim.mutex);for(int i=0;i<12;++i){
        r.pd_nm[i]=c.kp[i]*(c.target_rad[i]-sim.motors[i])-c.kd[i]*c.velocity_rad_s[i];
        r.friction_nm[i]=c.blocked_direction[i]?0:c.feedforward_nm[i];r.nominal_nm[i]=r.pd_nm[i]+c.feedforward_nm[i];}}
    r.nominal_nm=submit_glove(r.nominal_nm,deadline,true);return r;
}
void EncosDriver::disable() { std::lock_guard<std::mutex> l(sim.mutex); ++sim.glove_disable; sim.force_enabled=false; }
void EncosDriver::close() { std::lock_guard<std::mutex> l(sim.mutex); ++sim.glove_close; sim.glove_connected=false; }
}

struct Fixture {
    std::filesystem::path root, dir, mapping, linear, contact, recording;
    SessionOptions options;
    Vec20 initial{}, bent{};
    Fixture(const std::filesystem::path& project) : root(project) {
        char pattern[]="/tmp/motorhand-session-offline-XXXXXX";
        char* p=mkdtemp(pattern); check(p, "mkdtemp failed"); dir=p;
        mapping=dir/"mapping.json"; linear=dir/"linear.json"; contact=dir/"contact.json"; recording=dir/"trajectory.jsonl";
        json map{{"calibrated",true}, {"id_order_confirmed",true}, {"direction_confirmed",true}};
        std::ofstream(mapping)<<map.dump();
        json l; { std::ifstream f(root.parent_path()/"config/feedback/max-torque-linear-right.json"); f>>l; }
        l["wuji_serial_number"]=serial; l["source_id"]="offline-only-synthetic-limits";
        // A long local-Jacobian age isolates the asynchronous algorithm from
        // host scheduling; input frames and every command remain capped at 100ms.
        l["jacobian_max_age_s"]=5.0; l["jacobian_max_pose_delta_rad"]=1.0;
        l["ramp_duration_s"]=0.1; l["max_torque_slew_nm_s"]=std::vector<double>(12,100.0);
        std::ofstream(linear)<<l.dump();
        json c; { std::ifstream f(root.parent_path()/"config/feedback/contact-impedance-right.json"); f>>c; }
        c["source_id"]="offline-only-synthetic-contact";
        c["error_on_rad"]=std::vector<double>(20,0.005); c["error_off_rad"]=std::vector<double>(20,0.002);
        c["dwell_s"]=0.02; c["release_dwell_s"]=0.04;
        std::ofstream(contact)<<c.dump();
        options.asset_root=root.parent_path()/"assets/hand_motor_ve33";
        options.retarget_yaml=root/"config/retargeting/official/right.yaml";
        options.glove_mapping=mapping; options.linear_feedback=linear; options.contact_feedback=contact;
        options.wuji_serial=serial; options.approved_firmware=firmware; options.transition_ns=200000000;
        Ve33Model model(options.asset_root,false); WujiRetarget retarget(options.retarget_yaml,false);
        auto zero=model.zero(); initial=retarget.solve(zero.points_m);
        Vec12 m{}; m[4]=0.18; m[5]=0.18;
        auto pose=model.evaluate_motors(m,&zero); bent=retarget.solve(pose.points_m);
        check(distance(initial,bent)>0.005,"synthetic recorded trajectory must contain genuine motion");
        TrajectoryRecorder recorder(recording,"right",serial,firmware);
        TrajectoryFrame frame; frame.source_ns=1000000000; frame.position_rad=initial; frame.target_rad=initial;
        recorder.record(frame); frame.source_ns+=1000000000; frame.position_rad=bent; frame.target_rad=bent;
        recorder.record(frame); frame.source_ns+=1000000000; frame.position_rad=initial; frame.target_rad=initial;
        recorder.record(frame); recorder.finish();
    }
    ~Fixture() { std::error_code e; std::filesystem::remove_all(dir,e); }
};

void no_expired_commands() {
    std::lock_guard<std::mutex> lock(sim.mutex);
    for (const auto& w:sim.hand_writes) check(w.submitted<w.deadline,"expired Wuji command was emitted");
    for (const auto& w:sim.glove_writes) check(w.submitted<w.deadline,"expired glove command was emitted");
}

void replay_takeover_and_resume(const Fixture& f) {
    sim.reset(f.initial); sim.start();
    Trajectory trajectory(f.recording); Session session(f.options); session.connect();
    session.validate_trajectory(trajectory); session.enable();
    auto next=Clock::now(); auto begin=Clock::now();
    SessionTick last{}; int fresh_count=0, completed_glove=0;
    std::uint64_t previous_hand_stamp=0, previous_glove_stamp=0;
    int distinct_hand=0, distinct_glove=0;
    double replay_motion=0, glove_motion=0;
    std::vector<std::uint64_t> cycles;
    while (Clock::now()-begin < std::chrono::seconds(2)) {
        auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
        auto target=trajectory.at(elapsed); auto source=now_ns();
        auto tick=session.step_inference(target,source);
        check(tick.hand.received_ns && tick.glove.oldest_ns && tick.hand.received_ns<=tick.cycle_completed_ns && tick.glove.oldest_ns<=tick.cycle_completed_ns,"source time domain");
        check(tick.hand.received_ns>=previous_hand_stamp && tick.glove.oldest_ns>=previous_glove_stamp,"receive timestamps regressed");
        if (tick.hand.received_ns>previous_hand_stamp) ++distinct_hand;
        if (tick.glove.oldest_ns>previous_glove_stamp) ++distinct_glove;
        previous_hand_stamp=tick.hand.received_ns; previous_glove_stamp=tick.glove.oldest_ns;
        check(tick.wuji_submit_ns>=tick.cycle_started_ns && tick.wuji_submit_ns<tick.target_deadline_ns,"Wuji submission timing");
        if (tick.inverse_fresh) ++fresh_count;
        if (tick.glove_submit_ns) { ++completed_glove; check(tick.glove_submit_ns>=tick.glove_output_source_ns && tick.glove_submit_ns<tick.glove_output_source_ns+100000000,"glove completion provenance"); }
        replay_motion=std::max(replay_motion,distance(tick.wuji_target,f.initial));
        if (tick.glove_submit_ns) glove_motion=std::max(glove_motion,max_abs(tick.glove_target));
        cycles.push_back(tick.cycle_completed_ns-tick.cycle_started_ns); last=tick;
        next+=std::chrono::milliseconds(20); std::this_thread::sleep_until(next);
    }
    check(fresh_count>5 && completed_glove>3,"live inverse did not produce usable playback follow frames");
    check(replay_motion>0.001 && glove_motion>0.001,"replay must produce nontrivial Wuji and inverse glove targets");
    check(distinct_hand>40 && distinct_glove>40,"50Hz producer did not produce advancing physical receive timestamps");
    std::sort(cycles.begin(),cycles.end());
    check(cycles[cycles.size()*95/100]<20000000,"50Hz control blocked on simulated 20ms serial submission");
    Vec20 previous=last.wuji_target;
    Vec12 manual_start{};
    { std::lock_guard<std::mutex> lock(sim.mutex); sim.frozen=true; sim.manual=true; sim.loaded=true; sim.manual_motors=sim.motors; manual_start=sim.motors; }
    session.request_takeover(true);
    auto first=session.step_takeover();
    check(distance(first.wuji_target,previous)<0.03,"takeover first target discontinuity");
    auto take_start=Clock::now(); next=Clock::now();
    bool geometry=false, contact=false, torque=false, direction=false;
    while (Clock::now()-take_start<std::chrono::seconds(12)) {
        double t=std::chrono::duration<double>(Clock::now()-take_start).count();
        {
            std::lock_guard<std::mutex> lock(sim.mutex);
            const double a=std::min(1.0,t/0.6);
            sim.manual_motors[4]=manual_start[4]*(1-a)+0.18*a;
            sim.manual_motors[5]=manual_start[5]*(1-a)+0.18*a;
        }
        auto tick=session.step_takeover();
        geometry |= tick.force_geometry_valid && tick.jacobian_source_ns!=0;
        for (int i=0; i<20; ++i) if (tick.contact[i]) {
            contact=true;
            double error=tick.wuji_target[i]-tick.hand.position_rad[i];
            if (std::abs(tick.virtual_hand_nm[i])>1e-7 && tick.virtual_hand_nm[i]*error<0) direction=true;
        }
        if (tick.force_commanded && max_abs(tick.glove_target)>0.005) torque=true;
        if (geometry && contact && torque && direction) break;
        next+=std::chrono::milliseconds(20); std::this_thread::sleep_until(next);
    }
    check(geometry && contact && torque && direction,"real Jacobian/contact path did not produce opposing nonzero nominal torque");
    { std::lock_guard<std::mutex> lock(sim.mutex); sim.loaded=false; }
    bool released=false; next=Clock::now();
    for (int i=0; i<100; ++i) {
        auto tick=session.step_takeover();
        last=tick;
        if (tick.force_commanded && tick.force_geometry_valid &&
            std::none_of(tick.contact.begin(),tick.contact.end(),[](bool x){return x;}) && max_abs(tick.glove_target)<0.005) { released=true; break; }
        next+=std::chrono::milliseconds(20); std::this_thread::sleep_until(next);
    }
    check(released,"release did not remove contact and nominal torque");
    session.request_takeover(false);
    auto resumed=session.step_inference(trajectory.at(0),now_ns());
    check(distance(resumed.wuji_target,last.wuji_target)<0.03,"replay did not re-anchor to last takeover target");
    next=Clock::now(); bool followed=false;
    for (int i=0; i<100; ++i) {
        auto tick=session.step_inference(trajectory.at(std::uint64_t(i)*20000000),now_ns());
        followed |= tick.inverse_fresh && tick.glove_submit_ns;
        next+=std::chrono::milliseconds(20); std::this_thread::sleep_until(next);
    }
    check(followed,"replay did not reacquire inverse after takeover");
    session.stop(); no_expired_commands();
    { std::lock_guard<std::mutex> lock(sim.mutex); check(sim.hand_disable>0 && sim.glove_disable>0,"normal stop did not release both devices"); }
    sim.stop();
}

void fault_case(const Fixture& f, bool stale) {
    sim.reset(f.initial); sim.start(); Session s(f.options); s.connect(); s.enable();
    for (int i=0; i<10; ++i) { s.step_inference(f.initial,now_ns()); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
    if (stale) {
        { std::lock_guard<std::mutex> lock(sim.mutex); sim.stale_hand=true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        must_throw([&]{ s.step_inference(f.initial,now_ns()); },"stale hand frame must fault session");
    } else {
        { std::lock_guard<std::mutex> lock(sim.mutex); sim.fail_hand_send=true; }
        must_throw([&]{ s.step_inference(f.initial,now_ns()); },"send failure must fault session");
    }
    no_expired_commands();
    { std::lock_guard<std::mutex> lock(sim.mutex);
      check(sim.hand_disable>0 && sim.glove_disable>0 && sim.hand_close>0 && sim.glove_close>0,"fault must stop both devices"); }
    sim.stop();
}

void asynchronous_glove_fault(const Fixture& f) {
    sim.reset(f.initial); sim.start(); Session s(f.options); s.connect(); s.enable();
    s.request_takeover(true);
    { std::lock_guard<std::mutex> lock(sim.mutex); sim.fail_glove_send=true; }
    bool observed=false;
    auto end=Clock::now()+std::chrono::seconds(3);
    while (Clock::now()<end) {
        try { s.step_takeover(); }
        catch (const std::exception& e) {
            observed=std::string(e.what()).find("injected Encos send failure")!=std::string::npos;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    check(observed,"asynchronous Encos send failure did not propagate to session");
    no_expired_commands();
    { std::lock_guard<std::mutex> lock(sim.mutex);
      check(sim.hand_disable>0 && sim.glove_disable>0 && sim.hand_close>0 && sim.glove_close>0,
            "asynchronous glove fault must stop both devices"); }
    sim.stop();
}
void linear_feedback(const Fixture& f) {
    sim.reset(f.initial); sim.start();
    auto options = f.options; options.linear_force_feedback = true;
    Session session(options); session.connect(); session.enable();
    { std::lock_guard<std::mutex> lock(sim.mutex); sim.loaded = true; }
    session.request_takeover(true);
    bool delivered = false;
    auto end = Clock::now() + std::chrono::seconds(4);
    while (Clock::now() < end) {
        const auto tick = session.step_takeover();
        if (tick.force_commanded && tick.force_geometry_valid && tick.linear_force &&
            tick.contact_phase == ContactPhase::not_configured && max_abs(tick.glove_target) > .005) {
            delivered = true; break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    check(delivered, "linear feedback did not deliver a nonzero nominal torque");
    session.stop(); no_expired_commands(); sim.stop();
}
// Inference grasps (Wuji blocked at the open pose while the command is the bent pose);
// switching to takeover with the glove held still must keep the grasp command instead
// of falling back to the blocked position (the object would be released).
void grasp_handover(const Fixture& f) {
    sim.reset(f.initial); sim.start();
    Session session(f.options); session.connect(); session.enable();
    auto next=Clock::now();
    auto run_inference=[&](const Vec20& target, double seconds) {
        SessionTick tick{};
        auto begin=Clock::now();
        while (Clock::now()-begin<std::chrono::duration<double>(seconds)) {
            tick=session.step_inference(target,now_ns());
            next+=std::chrono::milliseconds(20); std::this_thread::sleep_until(next);
        }
        return tick;
    };
    run_inference(f.initial,0.6);
    { std::lock_guard<std::mutex> lock(sim.mutex); sim.frozen=true; }   // fingers blocked by the object
    auto grasp=run_inference(f.bent,1.2);
    check(distance(grasp.wuji_target,f.bent)<1e-6,"inference must command the grasp pose");
    { std::lock_guard<std::mutex> lock(sim.mutex); sim.manual=true; sim.manual_motors=sim.motors; }   // operator holds still
    session.request_takeover(true);
    auto first=session.step_takeover();
    check(distance(first.wuji_target,f.bent)<0.01,"takeover switch moved the hand");
    SessionTick last=first;
    next=Clock::now();
    for (int i=0; i<30; ++i) { last=session.step_takeover(); next+=std::chrono::milliseconds(20); std::this_thread::sleep_until(next); }
    const double span=distance(f.bent,f.initial);
    check(distance(last.wuji_target,f.bent)<0.3*span,"takeover released the grasp toward the blocked position");
    session.stop(); sim.stop(); no_expired_commands();
}

void zero_aligned_start(const Fixture& f) {
    sim.reset(f.bent);sim.start();
    auto options=f.options;options.zero_aligned_start=true;
    Session session(options);session.connect();session.enable();
    check(sim.hand_homes==1,"独立遥操必须显式回零Wuji一次");
    check(distance(session.read_wuji_state().position_rad,Vec20{})<1e-9,"开始跟随前Wuji必须回到自身零位");
    {std::lock_guard<std::mutex> l(sim.mutex);sim.manual=true;sim.manual_motors.fill(0);}
    session.request_takeover(true);
    SessionTick last{};auto next=Clock::now();
    auto run=[&](int frames){for(int i=0;i<frames;++i){last=session.step_takeover();next+=std::chrono::milliseconds(20);std::this_thread::sleep_until(next);}};
    run(25);
    check(distance(last.wuji_target,Vec20{})<.002,"手套零位必须保持Wuji零目标，不保留启动姿态");
    {std::lock_guard<std::mutex> l(sim.mutex);sim.manual_motors[4]=.18;sim.manual_motors[5]=.18;}
    run(25);
    check(distance(last.wuji_target,Vec20{})>.01,"手套运动必须产生非零跟随目标");
    {std::lock_guard<std::mutex> l(sim.mutex);sim.manual_motors.fill(0);}
    run(35);
    check(distance(last.wuji_target,Vec20{})<.005,"回到手套零位后不应产生随历史运动衰减的姿态偏置");
    check(sim.hand_homes==1,"正常跟随不能重复回零");
    session.stop();sim.stop();no_expired_commands();

    sim.reset(f.bent);sim.start();sim.fail_hand_home=true;
    Session failed(options);failed.connect();
    must_throw([&]{failed.enable();},"Wuji回零失败必须中止启动");
    check(sim.hand_writes.empty() && sim.hand_disable>0 && sim.glove_disable>0,
          "回零失败后不得进入跟随并须停止两侧输出");
    failed.stop();sim.stop();
}
void zero_aligned_saturation(const Fixture& f) {
    sim.reset(Vec20{});
    sim.hand_limit=.04;sim.start();
    auto options=f.options;options.zero_aligned_start=true;options.linear_force_feedback=true;
    Session session(options);session.connect();session.enable();
    {std::lock_guard<std::mutex> l(sim.mutex);sim.manual=true;sim.manual_motors.fill(0);}
    session.request_takeover(true);
    auto next=Clock::now();SessionTick last{};
    auto run=[&](int frames){for(int i=0;i<frames;++i){last=session.step_takeover();next+=std::chrono::milliseconds(20);std::this_thread::sleep_until(next);}};
    run(20);
    {std::lock_guard<std::mutex> l(sim.mutex);sim.manual_motors[4]=.3;sim.manual_motors[5]=.3;}
    run(35);
    int saturated=-1;
    for(int i=0;i<20;++i){check(std::abs(last.wuji_target[i])<=.040001,"零位偏移后仍须保持实际限位");
        if(std::abs(last.wuji_target[i])>=.03999)saturated=i;}
    check(saturated>=0,"测试必须实际触发平移后的目标限位");
    {std::lock_guard<std::mutex> l(sim.mutex);sim.loaded=true;sim.loaded_joint=saturated;}
    run(35);
    check(last.force_commanded && last.force_geometry_valid,"饱和测试必须得到有效力反馈计算");
    check(max_abs(last.glove_target)<.003,"仅饱和轴有电流时，截断行Jacobian不能产生手套反射力矩");
    session.stop();sim.stop();no_expired_commands();
}

void driver_impedance_session(const Fixture& f) {
    sim.reset(Vec20{});sim.start();auto options=f.options;options.zero_aligned_start=true;
    options.driver_impedance_feedback=true;options.driver_impedance_config=f.root.parent_path()/"config/feedback/wuji-driver-impedance-right.json";
    Session session(options);session.connect();session.enable();session.request_takeover(true);
    {std::lock_guard<std::mutex> l(sim.mutex);sim.manual=true;sim.frozen=true;sim.loaded=true;}
    bool seen_pd=false,seen_friction=false;auto next=Clock::now();
    for(int n=0;n<150;++n){
        {std::lock_guard<std::mutex> l(sim.mutex);sim.manual_motors[4]=std::min(.18,n*.003);sim.manual_motors[5]=std::min(.18,n*.003);}
        auto tick=session.step_takeover();
        if(tick.force_commanded){check(tick.driver_impedance,"Session必须保留驱动阻抗模式");
            check(tick.feedback_request_id&&tick.contact_evaluated_ns&&tick.impedance_planned_ns>=tick.contact_evaluated_ns&&
                  tick.glove_submit_ns>=tick.impedance_planned_ns,"Session必须将同一完成请求的判据/规划/SDK提交时刻带出");
            check(std::min(tick.feedback_input_hand.received_ns,*std::min_element(tick.glove_impedance_observation.source_ns.begin(),tick.glove_impedance_observation.source_ns.end()))==
                  tick.glove_output_source_ns,"Session代理源时刻必须匹配完成请求，不能借用当前UI帧");
            seen_pd|=max_abs(tick.glove_driver_kp)>0;seen_friction|=max_abs(tick.glove_friction_nm)>.005;
            if(max_abs(tick.glove_driver_kp)>0)for(int i=0;i<12;++i)if(tick.glove_driver_kp[i]>0)
                check(tick.glove_friction_nm[i]==0,"接触PD不能叠加摩擦");}
        if(seen_pd&&seen_friction)break;
        next+=std::chrono::milliseconds(20);std::this_thread::sleep_until(next);
    }
    check(seen_pd&&seen_friction,"真实Session/Jacobian/接触判据必须同时接通自由摩擦和接触驱动PD");
    session.stop();sim.stop();no_expired_commands();
}

int main(int argc, char** argv) {
    try {
        auto root=argc>1 ? std::filesystem::absolute(argv[1]) : std::filesystem::current_path();
        Fixture f(root); driver_impedance_session(f); zero_aligned_start(f); zero_aligned_saturation(f); replay_takeover_and_resume(f); grasp_handover(f); fault_case(f,true); fault_case(f,false); asynchronous_glove_fault(f); linear_feedback(f);
        std::cout<<"Offline native Session integration passed\n"; return 0;
    } catch (const std::exception& e) {
        sim.stop(); std::cerr<<"Offline native Session integration failed: "<<e.what()<<'\n'; return 1;
    }
}
