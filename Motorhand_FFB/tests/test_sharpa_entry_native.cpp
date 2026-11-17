#include "hands/sharpa/entry_options.hpp"
#include <iostream>
#include <stdexcept>
using namespace motorhand::native::sharpa;
void check(bool x){if(!x)throw std::runtime_error("Sharpa入口参数回归失败");}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed);}
int main(){try{
    auto sim=parse_options({"sim","--duration","1"});
    check(sim.simulation&&sim.fake_glove&&!sim.execute&&sim.duration_s==1);
    auto live=parse_options({"force-contact","--serial","EXPLICIT-SN","--execute"});
    check(live.execute&&live.serial=="EXPLICIT-SN"&&!live.simulation&&live.feedback);
    auto virtual_hand=parse_options({"sim-teleop"});
    check(virtual_hand.simulation&&!virtual_hand.fake_glove&&!virtual_hand.feedback);
    rejects([]{parse_options({"teleop","--rate","0"});});
    rejects([]{parse_options({"teleop","--hand","both"});});
    rejects([]{parse_options({"replay"});});
    rejects([]{parse_options({"teleop","--duration","nan"});});
    rejects([]{parse_options({"teleop","--unknown"});});
    std::cout<<"Sharpa入口参数回归通过\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
