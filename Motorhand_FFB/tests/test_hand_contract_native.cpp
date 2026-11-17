// 这些断言防止22轴被裁成20轴、负载单位混用、缓存轮询续期和跨身份反馈。
#include "common/hand.hpp"
#include <stdexcept>
#include <iostream>
using namespace motorhand::native;
void check(bool v) { if (!v) throw std::runtime_error("共同机械手契约断言失败"); }
template<class F> void rejects(F f) { bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected); }
template<std::size_t N> void contract() {
    HandDescriptor<N> d; d.identity={"fake", "offline", "test", false};
    d.online_source=HandStatusSource::complete_state_packet;d.enabled_source=HandStatusSource::device_report;
    d.load_unit=HandLoadUnit::newton_metre; d.load_source="fake-joint-torque";
    d.lower_rad.fill(-1);d.upper_rad.fill(1);
    for(std::size_t i=0;i<N;++i)d.joint_names[i]="joint_"+std::to_string(i);
    HandState<N> s;s.identity=d.identity;s.load_unit=d.load_unit;s.load_source=d.load_source;
    s.online_source=d.online_source;s.enabled_source=d.enabled_source;
    s.received_ns=1'000'000'000;s.connected=true;s.online.fill(true);s.has_load=true;
    s.position_rad[N-1]=.7;s.load[N-1]=.9;
    validate_hand_state(s,d,1'099'999'999,false);
    check(s.position_rad[N-1]==.7 && s.load[N-1]==.9);
    rejects([&]{validate_hand_state(s,d,1'100'000'000,false);});
    rejects([&]{validate_hand_state(s,d,999'999'999,false);});
    s.load_unit=HandLoadUnit::ampere;rejects([&]{validate_hand_state(s,d,1'000'000'001,false);});s.load_unit=d.load_unit;
    s.identity.model="other";rejects([&]{validate_hand_state(s,d,1'000'000'001,false);});s.identity=d.identity;
    s.online[N-1]=false;rejects([&]{validate_hand_state(s,d,1'000'000'001,false);});s.online[N-1]=true;
    s.load_source="other";rejects([&]{validate_hand_state(s,d,1'000'000'001,false);});s.load_source=d.load_source;
    rejects([&]{validate_hand_state(s,d,1'000'000'001,true);});
    s.enabled.fill(true);validate_hand_state(s,d,1'000'000'001,true);
    s.enabled_source=HandStatusSource::unavailable;rejects([&]{validate_hand_state(s,d,1'000'000'001,true);});s.enabled_source=d.enabled_source;
    s.online_source=HandStatusSource::per_joint;rejects([&]{validate_hand_state(s,d,1'000'000'001,true);});s.online_source=d.online_source;
    d.joint_names[N-1]=d.joint_names[0];rejects([&]{validate_hand_state(s,d,1'000'000'001,true);});
}
int main(){contract<20>();contract<22>();static_assert(std::tuple_size<RetargetJacobian<22>>::value==264);std::cout<<"20/22轴契约通过\n";}
