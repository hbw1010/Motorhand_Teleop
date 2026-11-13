#include "HandClientG4.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    gateway::RouteState route(3,1);auto packet=gateway::Frame(0x90,3,1);
    packet.len=8;packet.data[0]=0x40;packet.data[7]=100;
    check(route.Accept(packet,1'000'000),"位置反馈应可解码");
    auto first=route.TimedSnapshot(1'000'010);
    auto delayed=route.TimedSnapshot(1'500'000);
    check(first.position_received_us==1'000'000&&delayed.position_received_us==first.position_received_us,"延迟读取快照不能刷新真实位置收包时间");
    check(first.current_received_us==1'000'000 && delayed.current_received_us==1'000'000,"延迟快照不能刷新电流时间");
    check(delayed.state.samples==1&&delayed.state.age_ms==500,"保留原快照的计数和年龄");
    packet.len=6;packet.data[0]=0xa0;packet.data[1]=3;
    check(route.Accept(packet,1'600'000),"补充电流应可解码");
    check(route.TimedSnapshot(1'600'001).position_received_us==1'000'000,"仅电流更新不能冒充新位置包");
    auto current_only=route.TimedSnapshot(1'600'001);
    check(current_only.current_received_us==1'600'000 && current_only.position_received_us==1'000'000,"电流补包必须只更新电流时间");
    packet.data[1]=1;
    check(route.Accept(packet,1'650'000),"仅位置补包应可解码");
    auto position_only=route.TimedSnapshot(1'650'001);
    check(position_only.position_received_us==1'650'000 && position_only.current_received_us==1'600'000,"位置补包不能刷新电流时间");
    packet.len=8;packet.data[0]=0x40;packet.data[1]=0;
    check(route.Accept(packet,1'700'000),"第二位置反馈应可解码");
    auto next=route.TimedSnapshot(1'700'001);
    check(next.position_received_us==1'700'000&&next.state.samples==3,"新位置包必须保留解析时刻并递增样本");
    check(sizeof(HandSnapshot)==88,"旧Python C ABI快照布局不得改变");
    lqs::HandClientG4 client;
    check(client.GetTimedSnapshots().empty(),"未连接的时序快照应为空，不能制造收包时间");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
