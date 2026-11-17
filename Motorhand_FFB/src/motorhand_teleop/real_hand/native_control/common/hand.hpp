#pragma once
#include "types.hpp"
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace motorhand::native {
template<std::size_t N> using JointArray = std::array<double,N>;
template<std::size_t N> using RetargetJacobian = std::array<double,N*12>;
// 数组可承载广播状态，但必须明确它来自逐轴诊断、整手报告或完整状态包。
enum class HandStatusSource { unavailable, per_joint, device_report, complete_state_packet };
enum class HandLoadUnit { unavailable, ampere, newton_metre };
struct HandIdentity {
    std::string model, serial, firmware;
    bool left{};
    bool operator==(const HandIdentity& b) const {
        return model==b.model && serial==b.serial && firmware==b.firmware && left==b.left;
    }
};
template<std::size_t N> struct HandDescriptor {
    HandIdentity identity;
    std::array<std::string,N> joint_names;
    JointArray<N> lower_rad{}, upper_rad{};
    HandStatusSource online_source=HandStatusSource::unavailable;
    HandStatusSource enabled_source=HandStatusSource::unavailable;
    HandLoadUnit load_unit=HandLoadUnit::unavailable;
    std::string load_source;
};
template<std::size_t N> struct HandState {
    JointArray<N> position_rad{}, velocity_rad_s{}, load{};
    bool has_velocity{}, has_load{};
    std::uint64_t received_ns{}, device_time{}, sequence{};
    bool connected{}, fault{};
    std::array<bool,N> online{}, enabled{};
    HandIdentity identity;
    HandStatusSource online_source=HandStatusSource::unavailable;
    HandStatusSource enabled_source=HandStatusSource::unavailable;
    HandLoadUnit load_unit=HandLoadUnit::unavailable;
    std::string load_source;
};
inline constexpr std::uint64_t hand_timeout_ns=100'000'000;
template<std::size_t N> void validate_hand_state(const HandState<N>& s,const HandDescriptor<N>& d,
                                                std::uint64_t now,bool require_enabled) {
    if (!s.received_ns || s.received_ns>now || now-s.received_ns>=hand_timeout_ns)
        throw std::runtime_error("机械手原始反馈过期或来自未来");
    if (!(s.identity==d.identity) || d.identity.model.empty() || d.identity.serial.empty() ||
        !s.connected || s.fault) throw std::runtime_error("机械手身份或状态不匹配");
    if (s.online_source!=d.online_source ||
        (s.online_source!=HandStatusSource::per_joint && s.online_source!=HandStatusSource::complete_state_packet))
        throw std::runtime_error("机械手在线状态来源无效或不匹配");
    if ((s.enabled_source!=HandStatusSource::unavailable && s.enabled_source!=d.enabled_source) ||
        (require_enabled && s.enabled_source!=HandStatusSource::per_joint && s.enabled_source!=HandStatusSource::device_report))
        throw std::runtime_error("机械手使能状态来源无效或不匹配");
    if (s.has_load && (s.load_unit==HandLoadUnit::unavailable || s.load_unit!=d.load_unit ||
        s.load_source.empty() || s.load_source!=d.load_source)) throw std::runtime_error("机械手负载单位或来源不匹配");
    for (std::size_t i=0;i<N;++i) {
        if(d.joint_names[i].empty())throw std::runtime_error("机械手关节名缺失");
        for(std::size_t j=0;j<i;++j)if(d.joint_names[i]==d.joint_names[j])throw std::runtime_error("机械手关节名重复");
        if(!s.online[i] || (require_enabled && !s.enabled[i]) || !std::isfinite(s.position_rad[i]) ||
           !std::isfinite(d.lower_rad[i]) || !std::isfinite(d.upper_rad[i]) || d.lower_rad[i]>=d.upper_rad[i] ||
           (s.has_velocity && !std::isfinite(s.velocity_rad_s[i])) || (s.has_load && !std::isfinite(s.load[i])))
            throw std::runtime_error("机械手反馈或关节限位无效");
    }
}
template<std::size_t N> class HandModel {
public:
    virtual ~HandModel()=default;
    virtual JointArray<N> solve(const Points&)=0;
    virtual Points hand_points(const JointArray<N>&)=0;
    virtual std::array<double,21> task_scales() const=0;
    virtual void seed(const JointArray<N>&)=0;
    virtual void reset()=0;
    // 工作器在自己的线程中创建独立优化器；微分模式不共享控制先验。
    virtual std::unique_ptr<HandModel<N>> create_solver(bool differentiation=false) const=0;
    virtual void set_initial_hint(const JointArray<N>* hint)=0;
};
template<std::size_t N> using HandModelFactory=std::function<std::unique_ptr<HandModel<N>>() >;
template<std::size_t N> class HandDevice {
public:
    virtual ~HandDevice()=default;
    virtual void connect()=0;
    virtual HandState<N> read()=0;
    virtual const HandDescriptor<N>& descriptor() const=0;
    virtual void enable()=0;
    virtual void home(const std::atomic<bool>* stop_requested=nullptr)=0;
    virtual void send(const JointArray<N>&,std::uint64_t original_deadline_ns)=0;
    virtual void disable()=0;
    virtual void close()=0;
};
} // namespace motorhand::native
