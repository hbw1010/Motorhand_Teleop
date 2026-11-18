// 只读几何批处理，供比例拟合与误差验收；不创建设备、不发送控制。
#include "model.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
using namespace motorhand::native;
using Json=nlohmann::json;
int main(int argc,char** argv){
 try{
  std::filesystem::path root=MOTORHAND_PROJECT_ROOT,config;bool left=false;
  for(int i=1;i<argc;++i){const std::string key=argv[i];if(++i>=argc)throw std::invalid_argument("参数缺少值");
   const std::string value=argv[i];if(key=="--config")config=value;
   else if(key=="--hand"){if(value!="left"&&value!="right")throw std::invalid_argument("无效手侧");left=value=="left";}
   else if(key=="--root")root=value;else throw std::invalid_argument("未知参数: "+key);
  }
  sharpa::Model model(root/"assets/sharpa-hand",left,config);std::string line;
  while(std::getline(std::cin,line)){
   try{
    const auto row=Json::parse(line);const auto& input=row.at("points_m");
    if(!input.is_array()||input.size()!=21)throw std::invalid_argument("points_m需要21x3");
    Points points{};for(int i=0;i<21;++i){if(!input[i].is_array()||input[i].size()!=3)throw std::invalid_argument("points_m需要21x3");for(int k=0;k<3;++k)points[3*i+k]=input[i][k].get<double>();}
    if(row.value("reset",false))model.reset();
    if(row.contains("seed_rad"))model.seed(row.at("seed_rad").get<JointArray<22>>());
    const auto q=model.solve(points);const auto actual=model.hand_points(q);const auto physical=model.physical_hand_points(q);
    Json task=Json::array(),physical_points=Json::array();
    for(int i=0;i<21;++i){task.push_back({actual[3*i],actual[3*i+1],actual[3*i+2]});physical_points.push_back({physical[3*i],physical[3*i+1],physical[3*i+2]});}
    std::cout<<Json{{"q_rad",q},{"points_m",task},{"physical_points_m",physical_points},{"task_scales",model.task_scales()}}.dump()<<std::endl;
   }catch(const std::exception& e){std::cout<<Json{{"error",e.what()}}.dump()<<std::endl;}
  }
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
