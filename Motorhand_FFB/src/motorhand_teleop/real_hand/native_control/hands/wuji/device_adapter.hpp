#pragma once
#include "common/hand.hpp"
#include "hands/wuji/wuji_device.hpp"
namespace motorhand::native {
// 原C ABI保持20轴不变；只在此处把电流A与SDK状态转换为共同契约。
class WujiHandDevice final : public HandDevice<20> {
public:
    WujiHandDevice(const WujiIdentity& identity,const std::string& approved):device_(identity,approved) {
        descriptor_.identity={"wuji",identity.serial,identity.firmware,identity.left};
        descriptor_.online_source=descriptor_.enabled_source=HandStatusSource::per_joint;
        descriptor_.load_unit=HandLoadUnit::ampere;descriptor_.load_source="wuji-current";
        const char* fingers[]={"thumb","index","middle","ring","pinky"};
        for(int i=0;i<20;++i)descriptor_.joint_names[i]=std::string(fingers[i/4])+"_S"+std::to_string(i%4+1);
    }
    void connect() override {device_.connect();}
    const HandDescriptor<20>& descriptor() const override{return descriptor_;}
    HandState<20> read() override {
        const auto raw=device_.read();HandState<20> s;
        s.identity=descriptor_.identity;s.load_unit=descriptor_.load_unit;s.load_source=descriptor_.load_source;
        s.online_source=descriptor_.online_source;s.enabled_source=descriptor_.enabled_source;
        s.received_ns=raw.timestamp_ns;s.has_velocity=raw.has_velocity;s.has_load=raw.has_effort;
        s.connected=raw.phase==1||raw.phase==2;s.fault=!s.connected;
        for(int i=0;i<20;++i){s.position_rad[i]=raw.positions[i];s.velocity_rad_s[i]=raw.velocities_rad_s[i];
            s.load[i]=raw.effort_a[i];s.online[i]=raw.online[i];s.enabled[i]=raw.phase==2&&raw.enabled[i];
            descriptor_.lower_rad[i]=raw.lower_limits[i];descriptor_.upper_rad[i]=raw.upper_limits[i];}
        return s;
    }
    void enable() override{device_.enable();}
    void home(const std::atomic<bool>* stop=nullptr) override{device_.home(stop);}
    void send(const Vec20& q,std::uint64_t deadline) override{device_.send(q,deadline);}
    void disable() override{device_.disable();}
    void close() override{device_.close();}
private:
    WujiDevice device_;HandDescriptor<20> descriptor_;
};
}
