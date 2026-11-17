#include "simulation/fake_glove.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
namespace motorhand::native::simulation {
namespace {
using Clock=std::chrono::steady_clock;
constexpr double pi=3.14159265358979323846;
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
}
struct FakeGlove::Impl {
    const std::array<EncosMotorCalibration,12> cal;
    mutable std::mutex mutex;
    std::mutex close_mutex;
    std::condition_variable changed;
    std::thread producer;
    GloveFrame frame;
    Vec12 target{},nominal{};
    std::array<std::uint64_t,12> writes{};
    bool connected{},closed{},homed{},force_enabled{};
    explicit Impl(std::array<EncosMotorCalibration,12> c):cal(std::move(c)) {
        std::array<bool,12> seen{};
        for(const auto& m:cal) {
            const int raw=m.raw_motor_id-1;
            if(raw<0||raw>=12||seen[raw]||!std::isfinite(m.zero_deg)||!std::isfinite(m.sign*m.scale)||m.sign*m.scale==0||
                !std::isfinite(m.model_min_rad)||!std::isfinite(m.model_max_rad)||m.model_min_rad>0||m.model_max_rad<0||
                m.model_min_rad>=m.model_max_rad||(m.has_torque_limit&&(!std::isfinite(m.raw_torque_limit_nm)||m.raw_torque_limit_nm<=0)))
                throw std::invalid_argument("模拟手套校准无效");
            seen[raw]=true;
        }
    }
    void require_connected()const{if(!connected||closed)throw std::runtime_error("模拟手套未连接");}
    void require_output(bool force)const{require_connected();if(!homed||(force&&!force_enabled))throw std::runtime_error("模拟手套未取得对应输出权");}
    void validate_position(const Vec12& q)const {
        for(std::size_t i=0;i<12;++i)if(!std::isfinite(q[i])||q[i]<cal[i].model_min_rad||q[i]>cal[i].model_max_rad)
            throw std::invalid_argument("模拟手套目标超出标定限位");
    }
    void validate_torque(const Vec12& value)const {
        for(std::size_t i=0;i<12;++i)if(!std::isfinite(value[i])||!cal[i].has_torque_limit||
            std::abs(value[i])>cal[i].raw_torque_limit_nm/std::abs(cal[i].sign*cal[i].scale)+1e-10)
            throw std::invalid_argument("模拟手套名义力矩超出标定限额");
    }
    void produce() {
        std::unique_lock<std::mutex> lock(mutex);
        auto next=Clock::now();
        while(connected) {
            const auto stamp=now_ns();
            frame.motor_rad=target;
            for(std::size_t i=0;i<12;++i) {
                // 与EncosMotorCalibration相同的模型轴/原始编码器定义。
                frame.raw_degrees[i]=cal[i].zero_deg+target[i]*180/pi/(cal[i].sign*cal[i].scale);
                frame.received_ns[i]=stamp;
            }
            frame.oldest_ns=stamp;
            // 无测得电流；current_received_ns始终为0，不能将名义力矩伪装成反馈。
            changed.notify_all();
            next+=std::chrono::milliseconds(20);
            changed.wait_until(lock,next,[&]{return !connected;});
        }
    }
    void confirm_write(std::uint64_t deadline){
        const auto stamp=now_ns();
        if(!deadline||stamp>=deadline)throw std::runtime_error("模拟手套原始命令期限已过");
        writes.fill(stamp);
    }
};
FakeGlove::FakeGlove(std::array<EncosMotorCalibration,12> c):impl_(std::make_unique<Impl>(std::move(c))){}
FakeGlove::~FakeGlove(){close();}
void FakeGlove::connect(){
    auto& s=*impl_;std::unique_lock<std::mutex> lock(s.mutex);
    if(s.connected||s.closed)throw std::runtime_error("模拟手套不能重复连接");
    s.connected=true;
    try{s.producer=std::thread([&s]{s.produce();});}catch(...){s.connected=false;throw;}
    s.changed.wait(lock,[&]{return s.frame.oldest_ns!=0;});
}
GloveFrame FakeGlove::read(){auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.require_connected();return s.frame;}
std::array<EncosMotorCalibration,12> FakeGlove::calibration()const{return impl_->cal;}
std::array<std::uint64_t,12> FakeGlove::last_command_write_ns()const{std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->writes;}
void FakeGlove::set_manual_target(const Vec12& q){auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.require_connected();s.validate_position(q);s.target=q;}
void FakeGlove::home(){
    auto& s=*impl_;std::unique_lock<std::mutex> lock(s.mutex);s.require_connected();
    s.target={};s.nominal={};s.force_enabled=false;
    s.changed.wait(lock,[&]{return !s.connected||s.frame.motor_rad==Vec12{};});
    s.require_connected();s.homed=true;
}
void FakeGlove::enable_force(){auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.require_output(false);s.force_enabled=true;}
Vec12 FakeGlove::write_position(const Vec12& q,std::uint64_t deadline){
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.require_output(false);s.validate_position(q);s.confirm_write(deadline);
    s.target=q;s.force_enabled=false;s.nominal={};return q;
}
Vec12 FakeGlove::write_torque(const Vec12& tau,std::uint64_t deadline){
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.require_output(true);s.validate_torque(tau);s.confirm_write(deadline);s.nominal=tau;return tau;
}
DriverImpedanceSubmission FakeGlove::write_impedance(const DriverImpedanceCommand& c,std::uint64_t deadline){
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.require_output(true);s.validate_position(c.target_rad);
    DriverImpedanceSubmission out;out.command=c;out.estimate_received_ns=s.frame.received_ns;
    for(std::size_t i=0;i<12;++i){
        if(!std::isfinite(c.kp[i])||c.kp[i]<0||!std::isfinite(c.kd[i])||c.kd[i]<0||!std::isfinite(c.velocity_rad_s[i]))
            throw std::invalid_argument("模拟手套阻抗参数无效");
        out.pd_nm[i]=c.kp[i]*(c.target_rad[i]-s.frame.motor_rad[i])-c.kd[i]*c.velocity_rad_s[i];
        out.friction_nm[i]=c.feedforward_nm[i];out.nominal_nm[i]=out.pd_nm[i]+c.feedforward_nm[i];
    }
    s.validate_torque(out.nominal_nm);s.confirm_write(deadline);s.nominal=out.nominal_nm;return out;
}
Vec12 FakeGlove::last_nominal_nm()const{std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->nominal;}
void FakeGlove::disable(){auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.nominal={};s.force_enabled=false;}
void FakeGlove::close(){
    auto& s=*impl_;std::lock_guard<std::mutex> close_lock(s.close_mutex);
    {std::lock_guard<std::mutex> lock(s.mutex);s.connected=false;s.closed=true;s.nominal={};s.force_enabled=false;}
    s.changed.notify_all();if(s.producer.joinable())s.producer.join();
}
}
