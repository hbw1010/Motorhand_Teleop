#include "wuji_device.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <cstring>
#include <iostream>
using namespace motorhand::native;
struct RealHand {};
std::atomic<bool> entered{false}, disabled{false};
extern "C" {
int real_hand_abi_version(){return 6;}
RealHand* real_hand_create(const RealHandOptions*,char*,size_t){return new RealHand;}
void real_hand_destroy(RealHand* h){delete h;}
int real_hand_connect(RealHand*,char*,size_t){return 0;}
int real_hand_read_state(RealHand*,RealHandState*,char*,size_t){return 0;}
int real_hand_enable(RealHand*,char*,size_t){return 0;}
int real_hand_send(RealHand*,const double*,size_t,uint64_t,char*,size_t){return 0;}
int real_hand_close(RealHand*,char*,size_t){return 0;}
int real_hand_disable(RealHand*,char*,size_t){disabled.store(true);return 0;}
int real_hand_home(RealHand*,char* error,size_t n){
 entered.store(true);
 auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(300);
 while(!disabled.load() && std::chrono::steady_clock::now()<until)std::this_thread::sleep_for(std::chrono::milliseconds(1));
 if(n){std::strncpy(error,"离线模拟回零中断",n-1);error[n-1]=0;}
 return -2;
}
}
int main(){
 WujiDevice hand({"offline","offline-address","offline-fw",false},"offline-fw");
 std::atomic<bool> cancel{false};
 std::thread request([&]{while(!entered.load())std::this_thread::yield();cancel.store(true);});
 bool failed=false;auto start=std::chrono::steady_clock::now();
 try{hand.home(&cancel);}catch(...){failed=true;}
 request.join();
 auto elapsed=std::chrono::steady_clock::now()-start;
 if(!failed||!disabled||elapsed>std::chrono::milliseconds(150)){
  std::cerr<<"回零取消未及时传递到原生失能入口\n";return 1;
 }
 std::cout<<"Wuji回零取消传播通过\n";return 0;
}
