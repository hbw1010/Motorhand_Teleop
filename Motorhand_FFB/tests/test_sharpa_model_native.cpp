#include "motorhand_teleop/real_hand/native_control/hands/sharpa/model.hpp"
#include "motorhand_teleop/real_hand/native_control/ve33_model.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <tinyxml2.h>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
using namespace motorhand::native;
using motorhand::native::sharpa::Model;
void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
template<class F> void throws(F&& f) { bool caught=false;try {f();}catch(const std::exception&){caught=true;}check(caught,"应拒绝非法输入"); }
double distance(const Points& a,const Points& b) { double n=0;for(int i=0;i<63;++i)n+=(a[i]-b[i])*(a[i]-b[i]);return std::sqrt(n); }
JointArray<22> pose(Model& m,double scale) {
 JointArray<22> q{}; for(int i=0;i<22;++i) {
  const auto& n=m.joint_names()[i];
  if(n.find("MCP_AA")!=std::string::npos || n.find("pinky_CMC")!=std::string::npos)continue;
  q[i]=std::clamp(scale*(.3+.025*i),m.lower_limits()[i]+.01,m.upper_limits()[i]-.01);
 } return q;
}

void asset_intersections(const std::filesystem::path& repo) {
 const auto source=repo/"Motorhand_FFB/assets/sharpa-hand";
 const auto fixture=repo/".cache/sharpa-model-test/fixture";
 std::filesystem::remove_all(fixture);
 const auto assets=fixture/"assets/sharpa-hand";
 const auto urdf_dir=assets/"wave_01/right_sharpa_wave";
 std::filesystem::create_directories(urdf_dir);
 std::filesystem::create_directories(fixture/"config/retargeting/sharpa");
 std::filesystem::copy_file(source/"wave_01/right_sharpa_wave/right_sharpa_wave.urdf",urdf_dir/"right_sharpa_wave.urdf");
 std::filesystem::copy_file(repo/"Motorhand_FFB/config/retargeting/sharpa/model.json",fixture/"config/retargeting/sharpa/model.json");
 tinyxml2::XMLDocument xml;
 check(xml.LoadFile((source/"right.xml").string().c_str())==tinyxml2::XML_SUCCESS,"读取限位测试资产");
 auto* scene=xml.FirstChildElement("mujoco");
 auto* control=scene->FirstChildElement("actuator")->FirstChildElement("position");
 control->DeleteAttribute("inheritrange");control->SetAttribute("ctrlrange","0 0.5");control->SetAttribute("forcerange","-1.2 1.5");
 tinyxml2::XMLElement* first_joint=nullptr;
 auto find_joint=[&](auto&& self,tinyxml2::XMLElement* e)->void {
  if(!e || first_joint)return;
  if(std::string(e->Name())=="joint") {first_joint=e;return;}
  for(auto* c=e->FirstChildElement();c;c=c->NextSiblingElement())self(self,c);
 };find_joint(find_joint,scene->FirstChildElement("worldbody"));
 check(first_joint!=nullptr,"查找MJCF物理关节");
 auto save=[&]{check(xml.SaveFile((assets/"right.xml").string().c_str())==tinyxml2::XML_SUCCESS,"写入限位测试资产");};
 save();Model limited(assets,false);
 check(limited.lower_limits()[0]==0 && limited.upper_limits()[0]==.5 && limited.effort_limits()[0]==1.2,"URDF/MJCF/执行器限位及力矩交集");

 control->SetAttribute("ctrlrange","-0.15 1.9");first_joint->SetAttribute("range","-0.1 1.2");save();
 Model joint_limited(assets,false);check(joint_limited.lower_limits()[0]==-.1 && joint_limited.upper_limits()[0]==1.2,"MJCF物理关节更严格时取交集");
 control->SetAttribute("ctrlrange","-0.2 2");first_joint->SetAttribute("range","-0.3 2.2");save();
 Model urdf_limited(assets,false);check(urdf_limited.lower_limits()[0]==-.1745 && urdf_limited.upper_limits()[0]==1.9199,"URDF更严格时取交集");
 first_joint->SetAttribute("range","-0.1745 1.9199");
 control->SetAttribute("ctrlrange","0.1 0.5");save();throws([&]{Model invalid_zero(assets,false);});
 control->SetAttribute("ctrlrange","3 4");save();throws([&]{Model empty_intersection(assets,false);});
 control->SetAttribute("ctrlrange","0 0.5");control->SetAttribute("joint","left_thumb_CMC_FE");save();throws([&]{Model mismatched_side(assets,false);});
 std::filesystem::remove_all(fixture);
}
int main(int argc,char**argv) {
 try {
 const auto repo=argc>1?std::filesystem::path(argv[1]):std::filesystem::current_path().parent_path();
 const auto root=repo/"Motorhand_FFB/assets/sharpa-hand";
 asset_intersections(repo);
 Model trailing_separator(std::filesystem::path(root.string()+"/"),false);
 check(trailing_separator.joint_names()[0]=="right_thumb_CMC_FE","资产根路径尾随分隔符");
 double max_solve_ms=0;
 for(bool left:{false,true}) {
  Model m(root,left); const auto& names=m.joint_names();
  check(std::set<std::string>(names.begin(),names.end()).size()==22,"22个唯一关节");
  check(names[0]==(left?"left_thumb_CMC_FE":"right_thumb_CMC_FE"),"SDK/URDF关节顺序");
  check(names[17]==(left?"left_pinky_CMC":"right_pinky_CMC"),"小指CMC顺序");
  JointArray<22> zero{};
  for(int i=0;i<22;++i)check(m.lower_limits()[i]<=0 && m.upper_limits()[i]>=0 && m.lower_limits()[i]<m.upper_limits()[i] && m.effort_limits()[i]>0,"合法零位和限位");
  auto p=m.hand_points(zero);check(std::abs(p[3*9+2]-.0989)<1e-10,"掌长canonical +Z");
  check((left?-1:1)*p[3*5+1]>0,"左右canonical宽度符号");

  // 独立Python Pinocchio按URDF名字生成的非对称22轴FK基准；包含全部固定轴实际角。
  JointArray<22> golden_q{};
  for(int i=0;i<22;++i) {
   golden_q[i]=names[i].find("MCP_AA")!=std::string::npos || names[i].find("CMC_AA")!=std::string::npos?-.05+.005*i:.03+.015*i;
   if(i==17)golden_q[i]=.12;
  }
  const std::array<double,15> expected=left?std::array<double,15>{
   .02356987082140255,-.1235382661487051,.1087419013290611,.02329982530540698,-.03236304155203148,.1972739800633285,.032851926776792,-.009999761654150537,.1966396318368808,.04425656960511953,.01229990149873884,.1856667832069804,.05771635982513648,.02994459815698186,.1717551707906835
  }:std::array<double,15>{
   .02357065693650747,.1235382663834636,.1087418646802933,.02329983076115967,.03236371250170463,.197273688641289,.032851926776792,.01000023834584946,.1966396318368808,.04425655576761624,-.01229934306661616,.1856668016219467,.05771624356789004,-.02994445296623827,.1717549248393403
  };
  auto golden=m.hand_points(golden_q);for(int f=0;f<5;++f)for(int k=0;k<3;++k)check(std::abs(golden[3*(4+4*f)+k]-expected[3*f+k])<2e-12,"命名关节映射独立FK基准");
  auto q=pose(m,.7);auto j=m.point_jacobian(q);
  const auto canonical=m.hand_points(q);for(int k=0;k<3;++k)check(std::abs(canonical[3+k]-.5*canonical[6+k])<1e-14,"共同thumb CMC辅助点为腕与MCP中点");
  for(int k=0;k<3;++k)for(int c=0;c<22;++c)check(std::abs(j[(3+k)*22+c]-.5*j[(6+k)*22+c])<1e-14,"thumb CMC辅助点Jacobian同步");
  for(int k=0;k<22;++k) {
   auto a=q,b=q;const double h=1e-6;
   // 包括solve固定的6轴，FK仍必须按实际反馈角计算。
   if(a[k]-h<m.lower_limits()[k])a[k]+=h,b[k]+=3*h;
   else if(b[k]+h>m.upper_limits()[k])a[k]-=3*h,b[k]-=h;
   else a[k]-=h,b[k]+=h;
   auto pa=m.hand_points(a),pb=m.hand_points(b);
   for(int r=0;r<63;++r)check(std::abs((pb[r]-pa[r])/(b[k]-a[k])-j[r*22+k])<2e-7,"解析FK Jacobian与中心差分");
  }
  auto scales=m.task_scales();auto target=m.hand_points(q);for(int i=0;i<63;++i)target[i]/=scales[i/3];
  auto diff=m.create_solver(true);auto result=diff->solve(target);
  check(distance(diff->hand_points(result),m.hand_points(q))<5e-4,"同源姿态任务重建");
  for(int i:{3,6,10,14,17,19})check(result[i]==0,"不可观测轴固定零");
  m.seed(q);auto reference=m.create_solver();
  auto bad=target;bad[10]=std::numeric_limits<double>::quiet_NaN();throws([&]{m.solve(bad);});
  throws([&]{m.solve(Points{});});
  check(distance(m.hand_points(m.solve(target)),reference->hand_points(reference->solve(target)))<1e-10,"非法输入不污染热启动");

  auto invalid_seed=q;invalid_seed[1]=std::numeric_limits<double>::quiet_NaN();throws([&]{m.seed(invalid_seed);});throws([&]{m.set_initial_hint(&invalid_seed);});

  // 真实动力学反馈可跨过命令软限位；FK保留真实角，内部seed只投影先验。
  invalid_seed=q;invalid_seed[0]=m.upper_limits()[0]+.001;const auto original_actual=invalid_seed;
  auto feedback_points=m.hand_points(invalid_seed);auto boundary=invalid_seed;boundary[0]=m.upper_limits()[0];
  check(distance(feedback_points,m.hand_points(boundary))>1e-6,"实际反馈FK不得裁成命令限位");
  auto feedback_j=m.point_jacobian(invalid_seed);auto feedback_a=invalid_seed,feedback_b=invalid_seed;feedback_a[0]-=1e-6;feedback_b[0]+=1e-6;
  auto feedback_pa=m.hand_points(feedback_a),feedback_pb=m.hand_points(feedback_b);
  for(int r=0;r<63;++r)check(std::abs((feedback_pb[r]-feedback_pa[r])/2e-6-feedback_j[r*22])<2e-7,"越界真实反馈Jacobian保留实际角");
  m.seed(invalid_seed);check(invalid_seed==original_actual,"seed不篡改实测输入");
  auto bounded_from_actual=m.solve(target);for(int i=0;i<22;++i)check(bounded_from_actual[i]>=m.lower_limits()[i]&&bounded_from_actual[i]<=m.upper_limits()[i],"实测seed内部投影后solve严格界内");

  diff->set_initial_hint(&q);
  for(int axis:{0,1,2,4,5,7,8,9,11,12,13,15,16,18,20,21}) {
   const double h=1e-4;auto qp=q,qm=q;qp[axis]+=h;qm[axis]-=h;
   auto pp=m.hand_points(qp),pm=m.hand_points(qm);for(int i=0;i<63;++i)pp[i]/=scales[i/3],pm[i]/=scales[i/3];
   diff->reset();auto plus=diff->solve(pp);diff->reset();auto minus=diff->solve(pm);
   check(std::abs((plus[axis]-minus[axis])/(2*h)-1)<.02,"同源微分正负方向与关节导数");
  }
  diff->set_initial_hint(nullptr);
  auto first=diff->solve(target);m.seed(pose(m,1.1));auto again=diff->solve(target);
  for(int i=0;i<22;++i)check(std::abs(first[i]-again[i])<1e-7,"微分求解器状态隔离与确定性");
  Ve33Model glove(repo/"assets/hand_motor_ve33",left);auto initial=glove.zero();
  m.reset();JointArray<22> previous_out{};
  for(int frame=0;frame<90;++frame) {
   Vec12 joints{};for(int i=0;i<12;++i)joints[i]=.2*(.5+.5*std::sin(frame*.12+i*.2));
   auto g=glove.evaluate_joints(joints,&initial);initial=g;
   const auto start=std::chrono::steady_clock::now();
   auto out=m.solve(g.points_m);
   max_solve_ms=std::max(max_solve_ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
   for(int i=0;i<22;++i) {
    check(std::isfinite(out[i]) && out[i]>=m.lower_limits()[i] && out[i]<=m.upper_limits()[i],"闭链手套轨迹22维有界");
    if(frame)check(std::abs(out[i]-previous_out[i])<.15,"连续轨迹无关节跳变");
   }
   previous_out=out;
  }
 }
 throws([&]{Model missing(root/"missing",false);});
 std::cout<<"Sharpa模型回归通过；轨迹帧求解最大 "<<max_solve_ms<<" ms\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
