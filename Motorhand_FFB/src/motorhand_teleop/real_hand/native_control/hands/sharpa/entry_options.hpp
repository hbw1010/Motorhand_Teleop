#pragma once
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace motorhand::native::sharpa {
struct EntryOptions {
    std::string command,serial,pn,firmware="3.0.10",feedback_mode="driver-impedance";
    std::filesystem::path project_root,trajectory,output,mapping;
    bool left=false,execute=false,simulation=false,fake_glove=false,feedback=false,object=true,help=false;
    int rate=50,port=8766;
    double duration_s=0,hardness=.3,takeover_at_s=-1;
};
inline EntryOptions parse_options(const std::vector<std::string>& args) {
    EntryOptions o;
    if(args.empty()||args[0]=="--help"||args[0]=="help"){o.help=true;return o;}
    o.command=args[0];
    const std::vector<std::string> commands={"discover","inspect","teleop","force-contact","force-linear","record","replay","takeover","sim","sim-teleop"};
    bool known=false;for(const auto& c:commands)known|=c==o.command;
    if(!known)throw std::invalid_argument("未知Sharpa命令");
    o.simulation=o.command=="sim"||o.command=="sim-teleop";
    o.fake_glove=o.command=="sim";
    o.feedback=o.command=="force-contact"||o.command=="force-linear"||o.command=="takeover";
    for(std::size_t i=1;i<args.size();++i){
        const auto& a=args[i];
        auto value=[&](){if(++i>=args.size())throw std::invalid_argument("参数缺少值: "+a);return args[i];};
        if(a=="--execute")o.execute=true;
        else if(a=="--simulation")o.simulation=true;
        else if(a=="--fake-glove")o.fake_glove=true;
        else if(a=="--feedback")o.feedback=true;
        else if(a=="--no-object")o.object=false;
        else if(a=="--help")o.help=true;
        else if(a=="--serial"||a=="--sharpa-serial")o.serial=value();
        else if(a=="--expected-pn")o.pn=value();
        else if(a=="--approved-firmware")o.firmware=value();
        else if(a=="--feedback-mode")o.feedback_mode=value();
        else if(a=="--hand"){auto v=value();if(v!="left"&&v!="right")throw std::invalid_argument("hand必须是left或right");o.left=v=="left";}
        else if(a=="--root")o.project_root=value();
        else if(a=="--mapping")o.mapping=value();
        else if(a=="--trajectory")o.trajectory=value();
        else if(a=="--output")o.output=value();
        else if(a=="--rate")o.rate=std::stoi(value());
        else if(a=="--curves-port")o.port=std::stoi(value());
        else if(a=="--duration")o.duration_s=std::stod(value());
        else if(a=="--feedback-hardness")o.hardness=std::stod(value());
        else if(a=="--takeover-at")o.takeover_at_s=std::stod(value());
        else if(!a.empty()&&a[0]!='-'&&o.trajectory.empty())o.trajectory=a;
        else throw std::invalid_argument("不支持的Sharpa参数: "+a);
    }
    if(o.rate<1||o.rate>200||o.port<0||o.port>65535||!std::isfinite(o.duration_s)||o.duration_s<0||
       !std::isfinite(o.hardness)||o.hardness<0||o.hardness>1||!std::isfinite(o.takeover_at_s)||o.takeover_at_s< -1)
        throw std::invalid_argument("Sharpa频率、端口、时长或硬度非法");
    if(o.fake_glove&&!o.simulation)throw std::invalid_argument("假手套仅用于仿真");
    if(o.feedback_mode!="driver-impedance"&&o.feedback_mode!="torque")throw std::invalid_argument("反馈模式必须是driver-impedance或torque");
    if((o.command=="replay"||o.command=="takeover")&&o.trajectory.empty())throw std::invalid_argument("回放/接管必须提供22轴轨迹");
    if(o.pn.empty())o.pn=o.left?"WaveSE-L-01":"WaveSE-R-01";
    return o;
}
}
