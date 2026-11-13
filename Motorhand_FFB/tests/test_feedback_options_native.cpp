#define main motorhand_cli_main
#include "../src/motorhand_teleop/real_hand/native_control/main.cpp"
#undef main
#include <iostream>
Args parsed(std::initializer_list<const char*> tokens){std::vector<char*> argv;for(auto s:tokens)argv.push_back(const_cast<char*>(s));return parse(argv.size(),argv.data());}
void check(bool x,const char* s){if(!x)throw std::runtime_error(s);}
int main(){try{
 auto a=parsed({"native","force-contact"});auto o=session_options(a);
 check(o.driver_impedance_feedback&&o.feedback_hardness==.3&&o.zero_aligned_start,"force-contact应默认阻抗及人工硬度0.3");
 o=session_options(parsed({"native","force-contact","--feedback-mode","torque"}));check(!o.driver_impedance_feedback,"显式torque应保留回退");
 o=session_options(parsed({"native","force-contact","--feedback-hardness",".7"}));check(o.feedback_hardness==.7,"人工硬度选项必须进入Session");
 o=session_options(parsed({"native","force-linear"}));check(!o.driver_impedance_feedback&&o.linear_force_feedback,"force-linear应保持旧力矩路径");
 bool bad=false;try{parsed({"native","force-contact","--feedback-hardness","nan"});}catch(const std::exception&){bad=true;}check(bad,"非有限硬度必须拒绝");
 bad=false;try{parsed({"native","force-linear","--feedback-mode","driver-impedance"});}catch(const std::exception&){bad=true;}check(bad,"线性模式不能静默忽略driver-impedance");
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
