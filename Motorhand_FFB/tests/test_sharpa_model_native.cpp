#include "motorhand_teleop/real_hand/native_control/hands/sharpa/model.hpp"
#include "motorhand_teleop/real_hand/native_control/common/force_jacobian.hpp"
#include "motorhand_teleop/real_hand/native_control/ve33_model.hpp"

#include <Eigen/Core>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <yaml-cpp/yaml.h>
#include <tinyxml2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace motorhand::native;
using motorhand::native::sharpa::Model;
void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
template<class F> void throws(F&& f) { bool caught=false;try {f();}catch(const std::exception&){caught=true;}check(caught,"应拒绝非法输入"); }
double distance(const Points& a,const Points& b) { double n=0;for(int i=0;i<63;++i)n+=(a[i]-b[i])*(a[i]-b[i]);return std::sqrt(n); }
void same_joints(const JointArray<22>& a,const JointArray<22>& b,const char* why) {
 for(int i=0;i<22;++i)check(std::abs(a[i]-b[i])<1e-10,why);
}
JointArray<22> pose(Model& model,double scale) {
 JointArray<22> q{};
 for(int i=0;i<22;++i)q[i]=std::clamp(scale*(.3+.025*i),model.lower_limits()[i]+.01,model.upper_limits()[i]-.01);
 return q;
}
std::array<std::string,22> semantic_names(bool left) {
 const std::string side=left?"left_":"right_";
 const std::array<const char*,22> suffix={"thumb_CMC_FE","thumb_CMC_AA","thumb_MCP_FE","thumb_MCP_AA","thumb_IP",
  "index_MCP_FE","index_MCP_AA","index_PIP","index_DIP","middle_MCP_FE","middle_MCP_AA","middle_PIP","middle_DIP",
  "ring_MCP_FE","ring_MCP_AA","ring_PIP","ring_DIP","pinky_CMC","pinky_MCP_FE","pinky_MCP_AA","pinky_PIP","pinky_DIP"};
 std::array<std::string,22> result;for(int i=0;i<22;++i)result[i]=side+suffix[i];return result;
}

// 独立读取派生URDF，按公开关节语义重排；不调用Model的FK或point_jacobian。
// 21点的15个任务点均读取pad固定frame，辅助thumb CMC按接口约定取中点。
class PadReference {
 pinocchio::Model model;
 std::unique_ptr<pinocchio::Data> data;
 std::array<int,22> joints{};
 std::array<pinocchio::FrameIndex,21> frames{};
 Eigen::Vector3d task_axis_scale=Eigen::Vector3d::Ones(),wrist_offset=Eigen::Vector3d::Zero();
public:
 PadReference(const std::filesystem::path& config,bool left) {
  const auto yaml=YAML::LoadFile(config.string());
  if(const auto geometry=yaml["geometry"]) {
   const auto scale=geometry["task_axis_scale"].as<std::array<double,3>>();
   const auto offset=geometry["wrist_offset_m"].as<std::array<double,3>>();
   for(int axis=0;axis<3;++axis) {
    check(std::isfinite(scale[axis])&&scale[axis]>0&&std::isfinite(offset[axis]),"独立任务几何配置必须有效");
    task_axis_scale[axis]=scale[axis];wrist_offset[axis]=offset[axis];
   }
  }
  auto urdf=std::filesystem::path(yaml["optimizer"]["urdf_path"].as<std::string>());
  if(urdf.is_relative())urdf=config.parent_path()/urdf;
  pinocchio::urdf::buildModel(urdf.lexically_normal().string(),model);
  check(model.nq==22,"独立pad模型必须为22轴");
  const auto names=semantic_names(left);
  for(int i=0;i<22;++i) {
   check(model.existJointName(names[i]),"独立pad模型缺少语义关节");
   joints[i]=model.joints[model.getJointId(names[i])].idx_q();
  }
  const std::string side=left?"left_":"right_";
  auto frame=[&](const std::string& name){check(model.existFrame(name),"独立pad模型缺少任务frame");return model.getFrameId(name);};
  frames[0]=frame(side+"hand_C_MC");frames[1]=frame(side+"thumb_CMC_VL");
  const std::array<const char*,5> fingers={"thumb","index","middle","ring","pinky"};
  const std::array<const char*,3> suffix={"pad_pip","pad_dip","pad_tip"};
  for(int finger=0;finger<5;++finger) {
   if(finger)frames[4*finger+1]=frame(side+fingers[finger]+"_MCP_VL");
   for(int segment=0;segment<3;++segment)frames[4*finger+2+segment]=frame(side+fingers[finger]+"_"+suffix[segment]);
  }
  data=std::make_unique<pinocchio::Data>(model);
 }
 Points physical_points(const JointArray<22>& actual) {
  Eigen::VectorXd q=Eigen::VectorXd::Zero(model.nq);
  for(int i=0;i<22;++i)q[joints[i]]=actual[i];
  pinocchio::forwardKinematics(model,*data,q);pinocchio::updateFramePlacements(model,*data);
  const Eigen::Vector3d wrist=data->oMf[frames[0]].translation();
  Points result{};
  for(int i=0;i<21;++i)for(int axis=0;axis<3;++axis)result[3*i+axis]=data->oMf[frames[i]].translation()[axis]-wrist[axis];
  for(int axis=0;axis<3;++axis)result[3+axis]=.5*(result[axis]+result[6+axis]);
  return result;
 }
 Points points(const JointArray<22>& actual) {
  auto result=physical_points(actual);
  for(int point=2;point<21;++point)for(int axis=0;axis<3;++axis)
   result[3*point+axis]=(result[3*point+axis]-wrist_offset[axis])/task_axis_scale[axis];
  for(int axis=0;axis<3;++axis) {
   result[axis]=0;
   result[3+axis]=.5*result[6+axis];
  }
  return result;
 }
};

struct TemporaryDirectory {
 std::filesystem::path path;
 TemporaryDirectory() {
  char pattern[]="/tmp/sharpa-model-test-XXXXXX";const auto* created=mkdtemp(pattern);
  check(created,"创建Sharpa离线测试目录失败");path=created;
 }
 ~TemporaryDirectory(){std::error_code error;std::filesystem::remove_all(path,error);}
};

void asset_intersections(const std::filesystem::path& repo) {
 const auto source=repo/"Motorhand_FFB/assets/sharpa-hand";
 TemporaryDirectory temporary;
 const auto fixture=temporary.path;
 const auto assets=fixture/"assets/sharpa-hand";
 const auto urdf_dir=assets/"wave_01/right_sharpa_wave";
 std::filesystem::create_directories(urdf_dir);
 std::filesystem::create_directories(fixture/"config/retargeting/sharpa");
 for(const auto* name:{"right_sharpa_wave.urdf","right_sharpa_wave_pad.urdf"})
  std::filesystem::copy_file(source/"wave_01/right_sharpa_wave"/name,urdf_dir/name);
 std::filesystem::copy_file(repo/"Motorhand_FFB/config/retargeting/sharpa/right.yaml",fixture/"config/retargeting/sharpa/right.yaml");
 tinyxml2::XMLDocument xml;
 check(xml.LoadFile((source/"right.xml").string().c_str())==tinyxml2::XML_SUCCESS,"读取限位测试资产");
 auto* scene=xml.FirstChildElement("mujoco");
 auto* control=scene->FirstChildElement("actuator")->FirstChildElement("position");
 control->DeleteAttribute("inheritrange");control->SetAttribute("ctrlrange","0 0.5");control->SetAttribute("forcerange","-1.2 1.5");
 tinyxml2::XMLElement* first_joint=nullptr;
 auto find_joint=[&](auto&& self,tinyxml2::XMLElement* e)->void {
  if(!e||first_joint)return;
  if(std::string(e->Name())=="joint"){first_joint=e;return;}
  for(auto* c=e->FirstChildElement();c;c=c->NextSiblingElement())self(self,c);
 };find_joint(find_joint,scene->FirstChildElement("worldbody"));
 check(first_joint!=nullptr,"查找MJCF物理关节");
 auto save=[&]{check(xml.SaveFile((assets/"right.xml").string().c_str())==tinyxml2::XML_SUCCESS,"写入限位测试资产");};
 save();Model limited(assets,false);
 check(limited.lower_limits()[0]==0&&limited.upper_limits()[0]==.5&&limited.effort_limits()[0]==1.2,"URDF/MJCF/执行器限位及力矩交集");
 control->SetAttribute("ctrlrange","-0.15 1.9");first_joint->SetAttribute("range","-0.1 1.2");save();
 Model joint_limited(assets,false);check(joint_limited.lower_limits()[0]==-.1&&joint_limited.upper_limits()[0]==1.2,"MJCF物理关节更严格时取交集");
 control->SetAttribute("ctrlrange","-0.2 2");first_joint->SetAttribute("range","-0.3 2.2");save();
 Model urdf_limited(assets,false);check(urdf_limited.lower_limits()[0]==-.1745&&urdf_limited.upper_limits()[0]==1.9199,"URDF更严格时取交集");
 first_joint->SetAttribute("range","-0.1745 1.9199");
 control->SetAttribute("ctrlrange","0.1 0.5");save();throws([&]{Model invalid_zero(assets,false);});
 control->SetAttribute("ctrlrange","3 4");save();throws([&]{Model empty_intersection(assets,false);});
 control->SetAttribute("ctrlrange","0 0.5");control->SetAttribute("joint","left_thumb_CMC_FE");save();throws([&]{Model mismatched_side(assets,false);});
}

void fk_and_feedback_geometry(Model& model,PadReference& reference,bool left) {
 check(model.joint_names()==semantic_names(left),"全部22轴必须严格保持SDK语义顺序");
 JointArray<22> zero{};
 for(int i=0;i<22;++i)check(model.lower_limits()[i]<=0&&model.upper_limits()[i]>=0&&model.lower_limits()[i]<model.upper_limits()[i]&&model.effort_limits()[i]>0,"合法零位和限位");
 const auto straight=model.physical_hand_points(zero);
 check(std::abs(straight[3*9+2]-.0989)<1e-10,"掌长canonical +Z");
 check((left?-1:1)*straight[3*5+1]>0,"左右canonical宽度符号");
 auto asymmetric=pose(model,.7);
 for(int i=0;i<22;++i)if(model.joint_names()[i].find("AA")!=std::string::npos)asymmetric[i]=-.05+.005*i;
 asymmetric[17]=.12;
 for(const auto& q:{zero,asymmetric}) {
  const auto physical=model.physical_hand_points(q),expected_physical=reference.physical_points(q);
  for(int row=0;row<63;++row)check(std::abs(physical[row]-expected_physical[row])<2e-12,"独立pad frame的全21点物理FK基准");
  const auto points=model.hand_points(q),expected=reference.points(q);
  for(int row=0;row<63;++row)check(std::abs(points[row]-expected[row])<2e-12,"按配置独立归一化的全21点任务FK基准");
  const auto jacobian=model.point_jacobian(q);
  for(int axis=0;axis<22;++axis) {
   auto plus=q,minus=q;plus[axis]+=1e-6;minus[axis]-=1e-6;
   const auto a=reference.points(plus),b=reference.points(minus);
   double column_norm=0;
   for(int row=0;row<63;++row) {
    check(std::abs((a[row]-b[row])/2e-6-jacobian[row*22+axis])<2e-7,"解析反馈Jacobian与独立FK中心差分");
    column_norm+=std::abs(jacobian[row*22+axis]);
   }
   check(column_norm>1e-4,"22个真实反馈轴均须保留几何影响");
  }
 }
 // 实测角越过命令限位时，FK和局部导数仍对应原始反馈，不能裁剪。
 auto actual=asymmetric;actual[0]=model.upper_limits()[0]+.001;
 auto boundary=actual;boundary[0]=model.upper_limits()[0];
 const auto points=model.hand_points(actual);
 check(distance(points,model.hand_points(boundary))>1e-6,"实际反馈FK不得裁成命令限位");
 check(distance(points,reference.points(actual))<2e-12,"越界反馈任务FK仍与独立模型一致");
 check(distance(model.physical_hand_points(actual),reference.physical_points(actual))<2e-12,"越界反馈物理FK仍与独立模型一致");
 const auto jacobian=model.point_jacobian(actual);
 auto plus=actual,minus=actual;plus[0]+=1e-6;minus[0]-=1e-6;
 const auto a=reference.points(plus),b=reference.points(minus);
 for(int row=0;row<63;++row)check(std::abs((a[row]-b[row])/2e-6-jacobian[row*22])<2e-7,"越界反馈导数保留实际角");
 auto invalid=actual;invalid[4]=std::numeric_limits<double>::infinity();
 throws([&]{model.physical_hand_points(invalid);});throws([&]{model.hand_points(invalid);});throws([&]{model.point_jacobian(invalid);});
}

void check_scales(Model& model,const YAML::Node& config) {
 const auto scales=model.task_scales();
 const std::array<const char*,5> fingers={"thumb","index","middle","ring","pinky"};
 check(scales[0]==1,"腕任务比例保持1");
 check(scales[1]==scales[2],"拇指辅助中点必须与MCP使用同一逆向比例");
 for(int finger=0;finger<5;++finger) {
  if(finger)check(scales[4*finger+1]==1,"四指MCP不应用段比例");
  for(int segment=0;segment<3;++segment)
   check(scales[4*finger+2+segment]==config["retarget"]["segment_scaling"][fingers[finger]][segment].as<double>(),"15个任务点使用配置中的独立比例");
 }
 const auto points=model.hand_points(pose(model,.7));
 for(int axis=0;axis<3;++axis)
  check(std::abs(points[3+axis]/scales[1]-.5*points[6+axis]/scales[2])<1e-12,"逆向目标保持拇指辅助点中点约束");
}
void check_solution(Model& model,const YAML::Node& config,const JointArray<22>& solution) {
 for(int axis=0;axis<22;++axis)
  check(std::isfinite(solution[axis])&&solution[axis]>=model.lower_limits()[axis]&&solution[axis]<=model.upper_limits()[axis],"求解输出严格处于机械手命令范围");
 for(const auto& fixed:config["retarget"]["fixed_joints"]) {
  const auto name=fixed.first.as<std::string>();
  const auto& names=model.joint_names();const auto it=std::find(names.begin(),names.end(),name);
  check(it!=names.end(),"固定轴必须属于22轴语义");
  check(solution[std::distance(names.begin(),it)]==fixed.second.as<double>(),"只按当前yaml检查显式固定轴");
 }
}

void optimizer_state(Model& model,const Points& target,const YAML::Node& config) {
 const auto seed=pose(model,.6);
 auto control=model.create_solver(),reference=model.create_solver();
 control->seed(seed);reference->seed(seed);control->set_initial_hint(&seed);reference->set_initial_hint(&seed);
 const auto original_target=target;
 auto invalid=target;invalid[10]=std::numeric_limits<double>::quiet_NaN();
 throws([&]{control->solve(invalid);});throws([&]{control->solve(Points{});});
 auto invalid_seed=seed;invalid_seed[1]=std::numeric_limits<double>::quiet_NaN();
 throws([&]{control->seed(invalid_seed);});throws([&]{control->set_initial_hint(&invalid_seed);});
 const auto result=control->solve(target);
 same_joints(result,reference->solve(target),"拒绝非法输入不能污染既有先验及初值hint");
 check(target==original_target,"求解不能修改调用方任务点");check_solution(model,config,result);
 control->set_initial_hint(nullptr);reference->set_initial_hint(nullptr);
 same_joints(control->solve(target),reference->solve(target),"清除hint后的热启动状态一致");

 auto actual=seed;actual[0]=model.upper_limits()[0]+.001;const auto actual_copy=actual;
 auto bounded=actual;bounded[0]=model.upper_limits()[0];
 control->seed(actual);reference->seed(bounded);
 same_joints(control->solve(target),reference->solve(target),"实际超调seed只投影内部先验");
 check(actual==actual_copy,"seed不能篡改实测输入");

 // norm_delta是有意保留的时序先验。比较同历史双实例，而非要求连续两次求解完全相同。
 auto differential=model.create_solver(true),differential_reference=model.create_solver(true);
 differential->seed(seed);differential_reference->seed(seed);
 differential->set_initial_hint(&seed);differential_reference->set_initial_hint(&seed);
 same_joints(differential->solve(target),differential_reference->solve(target),"微分初始状态一致");
 control->seed(pose(model,1.1));control->set_initial_hint(nullptr);control->solve(target);
 same_joints(differential->solve(target),differential_reference->solve(target),"控制求解器不能污染微分求解器状态");
 differential->reset();differential_reference->reset();
 same_joints(differential->solve(target),differential_reference->solve(target),"reset保留hint且清除先验");
}

void feedback_jacobian(const std::filesystem::path& glove_root,Model& model,bool left) {
 Ve33Model glove(glove_root,left);const auto zero=glove.zero();Vec12 joints{};joints.fill(.12);
 const auto source=glove.evaluate_joints(joints,&zero);
 const auto centre=glove.evaluate_tree(source.tree_rad);
 constexpr double step=.001;
 const std::vector<int> columns={11,0,5}; // 非自然顺序，包含末指及跨指列。
 auto control_reference=model.create_solver();
 model.seed(pose(model,.5));control_reference->seed(pose(model,.5));
 CommonForceJacobianColumns<22> worker(glove_root,[&model]{return model.create_solver(true);},left);
 const auto actual=worker.compute(source,columns,step);
 check(actual.size()==22*columns.size(),"反馈Jacobian必须保留22行和请求列顺序");
 auto centre_solver=model.create_solver(true);
 const auto hint=centre_solver->solve(centre.points_m);
 double magnitude=0,last_row=0;
 for(std::size_t col=0;col<columns.size();++col) {
  auto plus=centre.joint_rad,minus=centre.joint_rad;plus[columns[col]]+=step;minus[columns[col]]-=step;
  // 独立的正负求解器，不复用worker内部状态；闭链扰动沿同一物理装配分支。
  const auto plus_pose=glove.evaluate_joints(plus,&centre),minus_pose=glove.evaluate_joints(minus,&centre);
  auto plus_solver=model.create_solver(true),minus_solver=model.create_solver(true);
  plus_solver->set_initial_hint(&hint);minus_solver->set_initial_hint(&hint);
  const auto a=plus_solver->solve(plus_pose.points_m),b=minus_solver->solve(minus_pose.points_m);
  for(int row=0;row<22;++row) {
   const double expected=(a[row]-b[row])/(2*step);
   check(std::abs(actual[row*columns.size()+col]-expected)<1e-7,"反馈Jacobian须等于独立闭链正负扰动导数");
   magnitude+=std::abs(expected);if(row==21)last_row+=std::abs(expected);
  }
 }
 check(magnitude>.05&&last_row>1e-5,"反馈导数必须包含真实运动及第22轴响应");
 same_joints(model.solve(centre.points_m),control_reference->solve(centre.points_m),"反馈微分不能污染创建它的控制求解器");
 // worker必须由树状态重建任务点，不能信任旁路填入的points_m。
 auto altered=source;altered.points_m.fill(1000);
 const auto rebuilt=worker.compute(altered,columns,step);
 for(std::size_t i=0;i<actual.size();++i)check(std::abs(actual[i]-rebuilt[i])<1e-7,"反馈求导须由实际树重建任务输入");
}

// 固定的非单位测试变换保证覆盖三轴比例和腕偏移，不依赖拟合参数是否已更新。
void task_geometry_override(const std::filesystem::path& root,const std::filesystem::path& source_config,
                            bool left,const Points& target) {
 TemporaryDirectory temporary;
 auto config=YAML::LoadFile(source_config.string());
 auto urdf=std::filesystem::path(config["optimizer"]["urdf_path"].as<std::string>());
 if(urdf.is_relative())urdf=source_config.parent_path()/urdf;
 config["optimizer"]["urdf_path"]=urdf.lexically_normal().string();
 config["geometry"]["task_axis_scale"]=std::vector<double>{1.2,.8,1.1};
 config["geometry"]["wrist_offset_m"]=std::vector<double>{.003,-.002,.004};
 const auto path=temporary.path/"geometry.yaml";
 {std::ofstream output(path);output<<config;check(output.good(),"写入独立任务几何测试配置失败");}
 Model calibrated(root,left,path);PadReference reference(path,left);
 fk_and_feedback_geometry(calibrated,reference,left);
 const auto actual=pose(calibrated,.5);
 check(distance(calibrated.hand_points(actual),calibrated.physical_hand_points(actual))>.005,
       "非单位校准必须实际改变任务坐标");
 Model physical_baseline(root,left);
 check(distance(calibrated.physical_hand_points(actual),physical_baseline.physical_hand_points(actual))<2e-12,
       "任务几何校准不能改变物理pad坐标");
 auto cloned=calibrated.create_solver();Model independent(root,left,path);
 cloned->seed(actual);independent.seed(actual);
 same_joints(cloned->solve(target),independent.solve(target),"独立求解器必须保留调用方的geometry配置覆盖");
}

int main(int argc,char** argv) {
 try {
  const auto repo=argc>1?std::filesystem::path(argv[1]):std::filesystem::current_path().parent_path();
  const auto root=repo/"Motorhand_FFB/assets/sharpa-hand";
  asset_intersections(repo);
  Model trailing_separator(std::filesystem::path(root.string()+"/"),false);
  check(trailing_separator.joint_names()[0]=="right_thumb_CMC_FE","资产根路径尾随分隔符");
  double max_solve_ms=0;
  for(bool left:{false,true}) {
   Model model(root,left);
   const auto config_path=repo/"Motorhand_FFB/config/retargeting/sharpa"/(left?"left.yaml":"right.yaml");
   const auto config=YAML::LoadFile(config_path.string());PadReference reference(config_path,left);
   fk_and_feedback_geometry(model,reference,left);check_scales(model,config);
   Ve33Model glove(repo/"assets/hand_motor_ve33",left);auto initial=glove.zero();
   task_geometry_override(root,config_path,left,initial.points_m);
   optimizer_state(model,initial.points_m,config);
   feedback_jacobian(repo/"assets/hand_motor_ve33",model,left);
   model.reset();JointArray<22> previous{};
   for(int frame=0;frame<90;++frame) {
    Vec12 joints{};for(int i=0;i<12;++i)joints[i]=.2*(.5+.5*std::sin(frame*.12+i*.2));
    auto g=glove.evaluate_joints(joints,&initial);initial=g;
    const auto start=std::chrono::steady_clock::now();const auto result=model.solve(g.points_m);
    max_solve_ms=std::max(max_solve_ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    check_solution(model,config,result);
    if(frame)for(int axis=0;axis<22;++axis)if(std::abs(result[axis]-previous[axis])>=.15)
     throw std::runtime_error("连续轨迹跳变: frame="+std::to_string(frame)+" axis="+std::to_string(axis)+" delta="+std::to_string(result[axis]-previous[axis]));
    previous=result;
   }
  }
  throws([&]{Model missing(root/"missing",false);});
  std::cout<<"Sharpa模型回归通过；独立pad FK、22轴反馈导数、求解器隔离；轨迹帧求解最大 "<<max_solve_ms<<" ms\n";
  return 0;
 } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
