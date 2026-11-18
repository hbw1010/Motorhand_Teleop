#include "entry_options.hpp"
#include "motion.hpp"
#include "model.hpp"
#include "device.hpp"
#include "feedback.hpp"
#include "common/session.hpp"
#include "encos_driver.hpp"
#include "playback_control.hpp"
#include "simulation/fake_glove.hpp"
#include "simulation/sharpa_device.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
using namespace motorhand::native;
namespace sh= motorhand::native::sharpa;
namespace sim= motorhand::native::simulation;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace {
std::atomic<bool> stopped{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void signal_stop(int){stopped.store(true);}
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
std::string failure_text(std::exception_ptr error){try{std::rethrow_exception(error);}catch(const std::exception& e){return e.what();}catch(...){return "未知错误";}}
class Publisher {
    int fd_=-1;sockaddr_in address_{};
public:
    explicit Publisher(int port){if(!port)return;fd_=socket(AF_INET,SOCK_DGRAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if(fd_<0)throw std::runtime_error("无法创建只读遥测端口");address_.sin_family=AF_INET;address_.sin_port=htons(port);inet_pton(AF_INET,"127.0.0.1",&address_.sin_addr);}
    ~Publisher(){if(fd_>=0)::close(fd_);}
    void send(const Json& j){if(fd_<0)return;auto s=j.dump();if(s.size()>60000)return;
        sendto(fd_,s.data(),s.size(),MSG_DONTWAIT,reinterpret_cast<sockaddr*>(&address_),sizeof(address_));}
};
struct FeedbackWindow {
    std::size_t ticks{}, geometry_ticks{};
    std::array<std::size_t,12> impedance_ticks{};
    Vec12 peak_nm{};
    std::array<std::optional<double>,12> peak_current_a{};

    void observe(const ControlTick<22>& tick,const std::array<EncosMotorCalibration,12>& calibration) {
        if(!tick.takeover)return;
        ++ticks;
        geometry_ticks+=tick.force_geometry_valid;
        const auto now=tick.cycle_completed_ns;
        for(std::size_t i=0;i<calibration.size();++i){
            const auto& axis=calibration[i];
            const auto raw=axis.raw_motor_id-1;
            if(tick.force_commanded){
                peak_nm[raw]=std::max(peak_nm[raw],std::abs(tick.glove_target[i]*axis.sign*axis.scale));
                if(tick.glove_driver_kp[i]>0)++impedance_ticks[raw];
            }
            const auto received=tick.glove.current_received_ns[i];
            if(received&&received<=now&&now-received<hand_timeout_ns){
                const auto current=std::abs(tick.glove.measured_current_a[i]);
                peak_current_a[raw]=std::max(peak_current_a[raw].value_or(0),current);
            }
        }
    }

    int peak_axis(int first,int end) const {
        return std::distance(peak_nm.begin(),std::max_element(peak_nm.begin()+first,peak_nm.begin()+end));
    }

    void print_axis(std::ostream& out,const char* finger,int first,int end) const {
        const int raw=peak_axis(first,end);
        out<<finger<<" J"<<(raw+1)<<" "<<std::setprecision(2)<<peak_nm[raw]
           <<" Nm · "<<impedance_ticks[raw]<<" 帧 · ";
        if(peak_current_a[raw])out<<*peak_current_a[raw]<<" A";
        else out<<"无电流读数";
    }

    Json summary() const {
        Json axes=Json::object();
        for(int raw=0;raw<12;++raw){
            axes["J"+std::to_string(raw+1)]={{"名义峰值Nm",peak_nm[raw]},
                {"阻抗帧",impedance_ticks[raw]},
                {"实测电流峰值A",peak_current_a[raw]?Json(*peak_current_a[raw]):Json(nullptr)}};
        }
        return {{"接管帧",ticks},{"几何有效帧",geometry_ticks},
            {"全手峰值轴",peak_axis(0,12)+1},{"电机",axes}};
    }
};

void start_sdk_log(const std::filesystem::path& project_root) {
    try {
        const auto path=project_root/".cache"/"sharpa-logs"/("sdk-"+std::to_string(getpid())+".log");
        std::filesystem::create_directories(path.parent_path());
        sh::configure_sdk_file_logging(path.string());
        std::cout<<"Sharpa SDK 详情 "<<path.string()<<std::endl;
    } catch(const std::exception& error) {
        std::cerr<<"Sharpa SDK 日志文件不可用，保留终端日志。"<<error.what()<<'\n';
    }
}
HandDescriptor<22> describe(sh::Model& model,const sh::EntryOptions& a){
    HandDescriptor<22>d;d.identity={"sharpa",a.serial,a.firmware,a.left};
    d.joint_names=model.joint_names();d.lower_rad=model.lower_limits();d.upper_rad=model.upper_limits();
    d.load_unit=HandLoadUnit::newton_metre;d.load_source=a.simulation?sim::sharpa_contact_load_source:sh::torque_source;
    d.online_source=HandStatusSource::complete_state_packet;d.enabled_source=HandStatusSource::device_report;
    if(a.simulation){d.identity.serial=a.left?"SIM-SHARPA-LEFT":"SIM-SHARPA-RIGHT";d.identity.firmware="mujoco-4.0.0";}
    return d;
}
std::unique_ptr<HandDevice<22>> create_device(const sh::EntryOptions& a,const HandDescriptor<22>& d){
    if(a.simulation)return std::make_unique<sim::SharpaDevice>(a.project_root/"assets/sharpa-hand",d,a.object);
    if(a.serial.empty())throw std::invalid_argument("真机必须显式指定--serial；先运行sharpa-discover");
    sh::DeviceOptions o;o.serial=a.serial;o.left=a.left;o.expected_pn=a.pn;o.approved_firmware=a.firmware;
    return std::make_unique<sh::Device>(o,d);
}
Json state_json(const HandState<22>& s){
    return Json{{"robot","sharpa"},{"hand",s.identity.left?"left":"right"},{"serial",s.identity.serial},
        {"firmware",s.identity.firmware},{"position_rad",s.position_rad},{"velocity_rad_s",s.has_velocity?Json(s.velocity_rad_s):Json(nullptr)},
        {"load_nm",s.has_load?Json(s.load):Json(nullptr)},{"load_source",s.load_source},
        {"received_ns",s.received_ns},{"sequence",s.sequence},{"fault",s.fault},
        {"enabled",std::all_of(s.enabled.begin(),s.enabled.end(),[](bool x){return x;})}};
}
int inspect_or_record(const sh::EntryOptions& a){
    sh::Model model(a.project_root/"assets/sharpa-hand",a.left);auto d=describe(model,a);auto device=create_device(a,d);
    try{
        device->connect();d=device->descriptor();
        if(a.command=="inspect"){std::cout<<state_json(device->read()).dump()<<'\n';device->close();return 0;}
        auto output=a.output.empty()?a.trajectory:a.output;
        if(output.empty())throw std::invalid_argument("record需要--output路径");
        sh::MotionRecorder recorder(output,d);const auto start=Clock::now();
        while(!stopped.load()){
            auto begin=Clock::now();auto s=device->read();validate_hand_state(s,d,now_ns(),false);recorder.append(s);
            if(a.duration_s>0&&std::chrono::duration<double>(Clock::now()-start).count()>=a.duration_s)break;
            std::this_thread::sleep_until(begin+std::chrono::nanoseconds(1000000000/a.rate));
        }
        device->close();recorder.finish();std::cout<<"已保存Sharpa实测轨迹："<<output<<'\n';return 0;
    }catch(...){auto original=std::current_exception();try{device->close();}catch(...){throw std::runtime_error(failure_text(original)+"；连接关闭未确认："+failure_text(std::current_exception()));}std::rethrow_exception(original);}
}
int run(const sh::EntryOptions& a){
    const bool replay=a.command=="replay"||a.command=="takeover";
    const bool output_hardware=!a.simulation||(!a.fake_glove&&(a.feedback||replay));
    if(output_hardware&&!a.execute)throw std::invalid_argument("此入口会控制真实设备；需显式添加--execute");
    if(!a.simulation)start_sdk_log(a.project_root);
    auto model=std::make_unique<sh::Model>(a.project_root/"assets/sharpa-hand",a.left);
    auto descriptor=describe(*model,a);auto device=create_device(a,descriptor);
    auto* simulation=dynamic_cast<sim::SharpaDevice*>(device.get());
    std::optional<sh::Motion> motion;
    if(replay)motion=sh::Motion::load(a.trajectory,descriptor);
    const auto mapping=a.mapping.empty()?a.project_root.parent_path()/"hardware/encos_dexhand/configs/sim/glove_mapping.json":a.mapping;
    const auto feedback_path=a.project_root/"config/feedback"/(a.left?"sharpa-left.json":"sharpa-right.json");
    const auto wuji_feedback=a.project_root.parent_path()/"config/feedback";
    const bool linear=a.command=="force-linear";
    EncosDriver calibration_only(mapping.string(),feedback_path.string());
    const auto calibration=calibration_only.calibration();
    std::unique_ptr<GloveDevice> fake;
    sim::FakeGlove* fake_ptr=nullptr;
    if(a.fake_glove){auto p=std::make_unique<sim::FakeGlove>(calibration);fake_ptr=p.get();fake=std::move(p);}
    ControlOptions o;o.asset_root=a.project_root.parent_path()/"assets/hand_motor_ve33";
    o.glove_mapping=mapping;o.left=a.left;o.linear_feedback=feedback_path;o.force_feedback=a.feedback;
    o.glove_output=a.feedback||replay;o.linear_force_feedback=linear;
    o.driver_impedance_feedback=a.feedback&&!o.linear_force_feedback&&a.feedback_mode=="driver-impedance";
    o.driver_impedance_config=wuji_feedback/"wuji-driver-impedance-right.json";
    o.driver_impedance_torque_margin=.95;
    // Sharpa使用已拟合的绝对任务坐标，不能再减去零位重定向结果。
    // 独立遥操用既有0.5秒过渡进入绝对目标；回放接管保留原抓握偏移语义。
    o.feedback_hardness=a.hardness;o.zero_aligned_start=false;
    if(replay) {
        // 手套相对上次求解位置累计移动超过约 0.11° 后更新目标。
        o.takeover_motor_deadband_rad=0.002;
    } else o.takeover_offset_motion_rad=0;
    o.home_hand_on_enable=true;o.stop_requested=&stopped;
    std::unique_ptr<HandFeedback<22>> feedback;
    if(a.feedback)feedback=std::make_unique<sh::Feedback>(feedback_path,wuji_feedback,linear,calibration,descriptor.load_source,replay?0:o.transition_ns);
    ControlSession<22> session(o,std::move(device),std::move(model),std::move(feedback),std::move(fake));
    Publisher publisher(a.port);TerminalKeys keys;
    try{
        const auto actual=session.connect();
        std::optional<sh::MotionRecorder> recorder;
        if(!a.output.empty())recorder.emplace(a.output,actual,sh::MotionTargetSource::submitted);
        std::filesystem::path scene_path=a.project_root/"assets/sharpa-hand"/(a.left?"left.xml":"right.xml");
        if(simulation){
            auto snapshot=simulation->snapshot();scene_path=a.project_root/".cache"/("sharpa-scene-"+std::to_string(getpid())+".xml");
            std::filesystem::create_directories(scene_path.parent_path());std::ofstream scene(scene_path);scene<<snapshot.scene_xml;
            if(!scene)throw std::runtime_error("无法保存只读仿真场景");
        }
        std::cout<<"启动回零："<<(a.simulation?"虚拟Sharpa":"真实Sharpa")
            <<(o.glove_output?(a.fake_glove?"与假手套":"与真实Motorhand"):"；Motorhand仅输入")<<std::endl;
        session.enable();bool takeover=!replay;session.request_takeover(takeover);
        const auto start=Clock::now();auto last_print=start;bool inverse_ready=false,automatic_takeover=false;
        FeedbackWindow feedback_window;
        PlaybackClock playback(motion?motion->duration_ns():1,now_ns());
        if(takeover)playback.pause(now_ns());
        std::cout<<(a.simulation?"Sharpa仿真":"Sharpa真机")<<"已开始；手侧="<<(a.left?"left":"right")
            <<"，反馈="<<(a.feedback?a.feedback_mode:"关闭")<<"。"
            <<(replay?"空格切换接管/回放，":"")<<"q或Ctrl+C退出。\n";
        while(!stopped.load()){
            const auto begin=Clock::now();const auto ns=now_ns();const double elapsed=std::chrono::duration<double>(begin-start).count();
            bool toggle=false;
            for(auto key=keys.read();key!=KeyEvent::none;key=keys.read()){if(key==KeyEvent::quit)stopped.store(true);if(key==KeyEvent::toggle)toggle=true;}
            if(stopped.load())break;
            if(replay&&a.takeover_at_s>=0&&!automatic_takeover&&elapsed>=a.takeover_at_s){toggle=true;automatic_takeover=true;}
            if(replay&&toggle){
                takeover=!takeover;
                if(takeover)playback.pause(ns,inverse_ready);
                else {playback.resume(ns);inverse_ready=false;}
                session.request_takeover(takeover);
            }
            if(fake_ptr&&takeover){
                const double fraction=.5-.5*std::cos(elapsed*2*3.141592653589793/6.0);
                const Vec12 raw_demo{.20,.15,.5,.5,.8,1.2,.8,1.2,.8,1.2,.8,1.2};Vec12 target{};
                for(int i=0;i<12;++i)target[i]=std::clamp(raw_demo[calibration[i].raw_motor_id-1]*fraction,calibration[i].model_min_rad,calibration[i].model_max_rad);
                fake_ptr->set_manual_target(target);
            }
            const auto position=playback.position(ns,inverse_ready);
            auto tick=takeover?session.step_takeover():session.step_inference(motion->at(position),ns);
            inverse_ready=tick.inverse_fresh;
            if(recorder)recorder->append(tick.hand,tick.hand_target);
            if(a.feedback)feedback_window.observe(tick,calibration);
            auto j=state_json(tick.hand);j["schema_version"]=2;j["joint_names"]=actual.joint_names;j["hand_target_rad"]=tick.hand_target;
            j["glove_motor_rad"]=tick.glove.motor_rad;j["takeover"]=tick.takeover;
            j["glove_tree_order"]="ve33_closure_independent_dependent";
            j["glove_tree_rad"]=tick.glove_pose.tree_rad;
            j["glove_received_ns"]=tick.glove.oldest_ns;
            j["glove_nominal_nm"]=tick.takeover&&tick.force_commanded?Json(tick.glove_target):Json(nullptr);
            j["glove_motor_target_rad"]=!tick.takeover&&tick.glove_submit_ns?Json(tick.glove_target):Json(nullptr);
            j["inverse_fresh"]=!tick.takeover&&tick.inverse_fresh;
            j["hand_submit_ns"]=tick.hand_submit_ns;j["glove_submit_ns"]=tick.glove_submit_ns;
            j["jacobian_source_ns"]=tick.jacobian_source_ns;
            j["force_commanded"]=tick.force_commanded;j["force_geometry_valid"]=tick.force_geometry_valid;
            j["contact"]=tick.contact;j["scene_path"]=scene_path.string();j["simulated"]=a.simulation;
            if(simulation){auto s=simulation->snapshot();j["qpos"]=s.qpos;j["simulation_time"]=s.simulation_time;j["contact_count"]=s.contact_count;}
            if(std::chrono::duration<double>(begin-last_print).count()>=1){
                std::ostringstream status;
                status<<std::fixed<<std::setprecision(1)
                      <<"t="<<elapsed<<"s mode="<<(replay?(takeover?"接管":"回放"):"遥操");
                if(feedback_window.ticks){
                    const int peak=feedback_window.peak_axis(0,12);
                    j["feedback_window"]=feedback_window.summary();
                    status<<" | 几何 "<<feedback_window.geometry_ticks<<'/'<<feedback_window.ticks
                          <<" 帧 | 名义峰值 J"<<peak+1<<" "
                          <<std::setprecision(2)<<feedback_window.peak_nm[peak]<<" Nm\n  ";
                    feedback_window.print_axis(status,"拇指",0,4);
                    status<<" | ";
                    feedback_window.print_axis(status,"食指",4,6);
                } else if(!takeover) {
                    status<<" | 逆解"<<(tick.inverse_fresh?"有效":"等待");
                }
                feedback_window={};
                std::cout<<status.str()<<std::endl;
                last_print=begin;
            }
            publisher.send(j);
            if(a.duration_s>0&&elapsed>=a.duration_s)break;
            if(replay&&!takeover&&inverse_ready&&position>=motion->duration_ns())break;
            std::this_thread::sleep_until(begin+std::chrono::nanoseconds(1000000000/a.rate));
        }
        session.stop();if(recorder)recorder->finish();return 0;
    }catch(...){auto original=std::current_exception();try{session.stop();}catch(...){throw std::runtime_error(failure_text(original)+"；停止未确认："+failure_text(std::current_exception()));}std::rethrow_exception(original);}
}
}
int main(int argc,char** argv){
    std::signal(SIGINT,signal_stop);std::signal(SIGTERM,signal_stop);
    try{
        auto a=sh::parse_options(std::vector<std::string>(argv+1,argv+argc));
        if(a.help){std::cout<<"Sharpa: discover | inspect | teleop | force-contact | force-linear | record | replay | takeover | sim | sim-teleop\n"
            <<"--serial SN --hand right|left --approved-firmware VERSION --execute --feedback --duration 秒 --trajectory 文件 --output 文件 --curves-port 8766\n"
            <<"仿真默认无物体；--with-object可恢复抓握球。sharpa-viewer并排显示Motorhand与Sharpa。\n"
            <<"真机输出需--execute；sim不访问硬件；sim-teleop默认仅读取真实手套。\n";return 0;}
        if(a.project_root.empty())a.project_root=MOTORHAND_PROJECT_ROOT;
        a.project_root=std::filesystem::absolute(a.project_root);
        if(a.command=="discover"){
            auto transport=sh::make_sdk_transport();Json devices=Json::array();const auto until=Clock::now()+std::chrono::seconds(3);
            do{
                devices=Json::array();
                for(const auto& d:transport->discover())if(d.is_hand)devices.push_back({{"serial",d.serial},{"pn",d.pn},{"firmware",d.firmware},{"hand",d.left?"left":"right"}});
                if(!devices.empty()||stopped.load())break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }while(Clock::now()<until);
            std::cout<<devices.dump()<<'\n';return 0;
        }
        if(a.command=="inspect"||a.command=="record")return inspect_or_record(a);
        return run(a);
    }catch(const std::exception& e){std::cerr<<"Sharpa失败："<<e.what()<<'\n';return 1;}
}
