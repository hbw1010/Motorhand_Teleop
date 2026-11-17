#include "simulation/fake_glove.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace motorhand::native;
using namespace motorhand::native::simulation;
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"必须拒绝非法模拟操作");}
std::array<EncosMotorCalibration,12> calibration(){std::array<EncosMotorCalibration,12> c;for(int i=0;i<12;++i)c[i]={i+1,10,-1,2,-180,180,-1,1,10,true};return c;}
GloveFrame wait_sample(FakeGlove& g,std::uint64_t previous){for(int i=0;i<100;++i){auto f=g.read();if(f.oldest_ns>previous)return f;std::this_thread::sleep_for(std::chrono::milliseconds(1));}throw std::runtime_error("模拟生产线程未更新");}
int main(){try{
    FakeGlove g(calibration());rejects([&]{g.read();});g.connect();auto first=g.read();
    check(first.oldest_ns&&first.raw_degrees[0]==10,"初始零位必须使用标定原始角度");
    int cached=0;for(int i=0;i<1000;++i)if(g.read().oldest_ns==first.oldest_ns)++cached;
    check(cached>0,"read不能以轮询时刻续期");
    Vec12 manual{};manual[0]=.2;g.set_manual_target(manual);auto moved=wait_sample(g,first.oldest_ns);
    check(moved.motor_rad[0]==.2&&std::abs(moved.raw_degrees[0]-(10-.2*90/3.141592653589793))<1e-10,"模型与原始角转换必须保留zero/sign/scale");
    check(moved.current_received_ns[0]==0,"模拟名义输出不能伪装成测得电流");
    rejects([&]{g.write_position(manual,now_ns()+100'000'000);});g.home();
    auto zero=g.read();check(zero.motor_rad==Vec12{},"home必须在生产缓存确认归零后返回");
    g.write_position(manual,now_ns()+100'000'000);moved=wait_sample(g,zero.oldest_ns);check(moved.motor_rad==manual,"位置目标必须由生产线程跟随");
    g.enable_force();Vec12 torque{};torque[0]=.5;g.write_torque(torque,now_ns()+100'000'000);
    auto torque_frame=wait_sample(g,moved.oldest_ns);check(torque_frame.motor_rad==manual&&g.last_nominal_nm()[0]==.5,"力矩仅记录名义输出，不改变人手姿态");
    DriverImpedanceCommand c;c.target_rad=manual;c.target_rad[0]=.3;c.kp[0]=2;c.kd[0]=.5;c.velocity_rad_s[0]=.2;c.feedforward_nm[0]=.1;
    auto result=g.write_impedance(c,now_ns()+100'000'000);check(std::abs(result.nominal_nm[0]-.2)<1e-12,"阻抗名义输出必须使用同一缓存姿态");
    check(g.last_command_write_ns()[0]>0,"模拟输出必须记录实际提交时间");rejects([&]{g.write_torque(torque,now_ns()-1);});
    g.disable();check(g.last_nominal_nm()==Vec12{},"disable必须卸力");rejects([&]{g.write_torque(torque,now_ns()+100'000'000);});
    g.close();rejects([&]{g.read();});g.close();
    std::cout<<"假手套缓存、标定和名义输出通过\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
