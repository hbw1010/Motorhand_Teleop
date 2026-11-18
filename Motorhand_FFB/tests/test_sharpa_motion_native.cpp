#include "hands/sharpa/motion.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace motorhand::native;
using namespace motorhand::native::sharpa;
int main(){
    const auto file=std::filesystem::temp_directory_path()/("sharpa-motion-"+std::to_string(getpid())+".jsonl");
    try{
        HandDescriptor<22>d;d.identity={"sharpa","TEST","test",false};
        for(int i=0;i<22;++i){d.joint_names[i]="right_"+std::to_string(i);d.lower_rad[i]=-1;d.upper_rad[i]=1;}
        {MotionRecorder out(file,d);JointArray<22>a{},b{};b[21]=.6;out.append(1000,a);out.append(1000001000,b);}
        auto motion=Motion::load(file,d);auto q=motion.at(500000000);
        if(std::abs(q[21]-.3)>1e-12||motion.duration_ns()!=1000000000)throw std::runtime_error("22轴插值错误");
        d.identity.left=true;bool rejected=false;try{Motion::load(file,d);}catch(const std::exception&){rejected=true;}
        if(!rejected)throw std::runtime_error("错误手侧未拒绝");
        std::filesystem::remove(file);d.identity.left=false;
        {MotionRecorder out(file,d);JointArray<22>q{};q[21]=1.00004;out.append(1000,q);out.append(2000,q);}
        auto bounded=Motion::load(file,d);
        if(bounded.at(0)[21]!=1.0)throw std::runtime_error("实际微量越界不应成为非法回放目标");
        std::ifstream input(file);std::string line;std::getline(input,line);std::getline(input,line);
        auto raw=nlohmann::json::parse(line);
        if(raw.at("position_rad")[21]!=1.00004||raw.at("target_rad")[21]!=1.0)throw std::runtime_error("实测与回放目标必须分别保留");
        input.close();std::filesystem::remove(file);
        {
            MotionRecorder out(file,d,MotionTargetSource::submitted);
            HandState<22> s;s.received_ns=1000;s.position_rad[21]=.2;
            JointArray<22> submitted{};submitted[21]=.55;
            out.append(s,submitted);s.received_ns=2000;out.append(s,submitted);out.finish();
        }
        std::ifstream grasp(file);
        std::getline(grasp,line);
        if(nlohmann::json::parse(line).at("target_source")!="submitted_position_target")
            throw std::runtime_error("控制录制必须声明提交目标来源");
        std::getline(grasp,line);raw=nlohmann::json::parse(line);
        if(raw.at("position_rad")[21]!=.2||Motion::load(file,d).at(500)[21]!=.55)
            throw std::runtime_error("受阻录制应保留原始实测及回放所需的握紧目标");
        grasp.close();std::filesystem::remove(file);std::cout<<"Sharpa轨迹回归通过\n";
    }catch(const std::exception&e){std::filesystem::remove(file);std::cerr<<e.what()<<'\n';return 1;}
}
