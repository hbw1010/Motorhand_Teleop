#include "model.hpp"
#include "adaptive_retarget.hpp"
#include <Eigen/Core>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <yaml-cpp/yaml.h>
#include <tinyxml2.h>
#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace motorhand::native::sharpa {
namespace {
using V3=Eigen::Vector3d;
using V22=Eigen::Matrix<double,22,1>;
using J3=Eigen::Matrix<double,3,22>;
using J6=Eigen::Matrix<double,6,22>;
using Keypoints=std::array<V3,21>;
using Jacobians=std::array<J3,21>;
[[noreturn]] void fail(const std::string& s) {throw std::runtime_error("Sharpa模型："+s);}
std::string attr(const tinyxml2::XMLElement* e,const char* key) {
 if(!e || !e->Attribute(key))fail(std::string("缺少XML属性 ")+key);
 return e->Attribute(key);
}
double number(const tinyxml2::XMLElement* e,const char* key) {
 double v=0;if(!e || e->QueryDoubleAttribute(key,&v)!=tinyxml2::XML_SUCCESS || !std::isfinite(v))fail(std::string("非法XML数值 ")+key);return v;
}
std::array<double,2> range(const tinyxml2::XMLElement* e,const char* key) {
 std::istringstream in(attr(e,key));std::array<double,2> v;std::string extra;
 if(!(in>>v[0]>>v[1]) || in>>extra || !std::isfinite(v[0]) || !std::isfinite(v[1]) || v[0]>=v[1])fail("非法XML区间");return v;
}
void load_xml(tinyxml2::XMLDocument& xml,const std::filesystem::path& path) {
 if(xml.LoadFile(path.string().c_str())!=tinyxml2::XML_SUCCESS)fail("无法读取 "+path.string());
}
}
struct Model::Impl {
 std::filesystem::path root,config_path;
 bool left{};
 pinocchio::Model model;
 std::unique_ptr<pinocchio::Data> data;
 std::array<std::string,22> names;
 JointArray<22> lower{},upper{},effort{},solve_lower{},solve_upper{};
 std::array<int,22> semantic{};
 std::array<pinocchio::FrameIndex,21> frames{};
 std::unique_ptr<AdaptiveRetarget> optimizer;
 V3 task_axis_scale=V3::Ones(),wrist_offset=V3::Zero();

 Impl(std::filesystem::path path,bool is_left,std::filesystem::path config)
     :root(std::filesystem::canonical(path)),config_path(std::move(config)),left(is_left) {
  const std::string side=left?"left":"right";
  if(config_path.empty())config_path=root.parent_path().parent_path()/"config/retargeting/sharpa"/(side+".yaml");
  config_path=std::filesystem::absolute(config_path);
  const auto cfg=YAML::LoadFile(config_path.string());
  if(const auto geometry=cfg["geometry"]){
   const auto scales=geometry["task_axis_scale"].as<std::vector<double>>();
   const auto offset=geometry["wrist_offset_m"].as<std::vector<double>>();
   if(scales.size()!=3||offset.size()!=3)fail("几何校准需要3轴比例和腕偏移");
   for(int k=0;k<3;++k){
    if(!std::isfinite(scales[k])||scales[k]<=0||!std::isfinite(offset[k]))fail("几何校准必须有限且比例为正");
    task_axis_scale[k]=scales[k];wrist_offset[k]=offset[k];
   }
  }
  const auto urdf=root/"wave_01"/(side+"_sharpa_wave")/(side+"_sharpa_wave.urdf");
  tinyxml2::XMLDocument xml;load_xml(xml,urdf);
  auto* robot=xml.FirstChildElement("robot");if(!robot)fail("无效URDF");
  std::set<std::string> unique;int count=0;
  for(auto* e=robot->FirstChildElement("joint");e;e=e->NextSiblingElement("joint")) {
   if(attr(e,"type")=="fixed")continue;
   if(attr(e,"type")!="revolute" || count>=22)fail("URDF应恰有22个转动关节");
   const auto n=attr(e,"name");if(n.rfind(side+"_",0)!=0 || !unique.insert(n).second)fail("关节侧别错误或重复");
   names[count]=n;const auto* limit=e->FirstChildElement("limit");
   lower[count]=number(limit,"lower");upper[count]=number(limit,"upper");effort[count]=number(limit,"effort");++count;
  }
  if(count!=22)fail("URDF关节数不是22");
  tinyxml2::XMLDocument mj;load_xml(mj,root/(side+".xml"));
  const auto* scene=mj.FirstChildElement("mujoco");if(!scene)fail("无效MJCF");
  if(attr(scene->FirstChildElement("compiler"),"angle")!="radian")fail("MJCF必须使用弧度");
  std::unordered_map<std::string,const tinyxml2::XMLElement*> mj_joints,actuators;
  auto walk=[&](auto&& self,const tinyxml2::XMLElement* parent)->void {
   if(!parent)return;
   for(const auto* e=parent->FirstChildElement();e;e=e->NextSiblingElement()) {
    if(std::string(e->Name())=="joint") {if(!mj_joints.emplace(attr(e,"name"),e).second)fail("MJCF关节重复");}
    self(self,e);
   }
  };walk(walk,scene->FirstChildElement("worldbody"));
  const auto* actuator=scene->FirstChildElement("actuator");if(!actuator)fail("缺少MJCF执行器");
  for(const auto* e=actuator->FirstChildElement();e;e=e->NextSiblingElement()) {
   if(std::string(e->Name())!="position" || !actuators.emplace(attr(e,"joint"),e).second)fail("无效或重复的MJCF位置执行器");
  }
  if(mj_joints.size()!=22 || actuators.size()!=22)fail("MJCF关节/执行器数不是22");
  for(int i=0;i<22;++i) {
   if(!mj_joints.count(names[i]) || !actuators.count(names[i]))fail("MJCF缺少命名关节 "+names[i]);
   const auto joint_range=range(mj_joints.at(names[i]),"range");const auto* a=actuators.at(names[i]);
   auto ctrl=joint_range;
   if(a->Attribute("ctrlrange"))ctrl=range(a,"ctrlrange");
   else {const double factor=number(a,"inheritrange");if(factor<=0)fail("非法inheritrange");const double mid=(ctrl[0]+ctrl[1])*.5,half=(ctrl[1]-ctrl[0])*.5*factor;ctrl={mid-half,mid+half};}
   lower[i]=std::max({lower[i],joint_range[0],ctrl[0]});upper[i]=std::min({upper[i],joint_range[1],ctrl[1]});
   if(a->Attribute("forcerange")) {const auto force=range(a,"forcerange");effort[i]=std::min({effort[i],-force[0],force[1]});}
   if(!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>=upper[i] || lower[i]>0 || upper[i]<0 || !std::isfinite(effort[i]) || effort[i]<=0)fail("无效限位交集或零位不合法："+names[i]);
  }
  auto task_urdf=std::filesystem::path(cfg["optimizer"]["urdf_path"].as<std::string>());
  if(task_urdf.is_relative())task_urdf=config_path.parent_path()/task_urdf;
  pinocchio::urdf::buildModel(task_urdf.lexically_normal().string(),model);
  if(model.nq!=22 || model.nv!=22)fail("Pinocchio维度不是22");
  for(int i=0;i<22;++i) {
   if(!model.existJointName(names[i]))fail("Pinocchio缺少关节");
   const auto& j=model.joints[model.getJointId(names[i])];
   if(j.nq()!=1 || j.nv()!=1 || j.idx_q()!=j.idx_v())fail("Pinocchio关节必须为标量");semantic[i]=j.idx_q();
  }
  auto frame=[&](const std::string& name) {if(!model.existFrame(name))fail("缺少任务frame "+name);return model.getFrameId(name);};
  frames[0]=frame(side+"_hand_C_MC");
  frames[1]=frame(side+"_thumb_CMC_VL");frames[2]=frame(side+"_thumb_pad_pip");frames[3]=frame(side+"_thumb_pad_dip");frames[4]=frame(side+"_thumb_pad_tip");
  const std::array<std::string,4> fingers={"index","middle","ring","pinky"};
  const std::array<std::string,4> suffix={"MCP_VL","pad_pip","pad_dip","pad_tip"};
  for(int f=0;f<4;++f)for(int k=0;k<4;++k)frames[5+4*f+k]=frame(side+"_"+fingers[f]+"_"+suffix[k]);
  data=std::make_unique<pinocchio::Data>(model);
  solve_lower=lower;solve_upper=upper;
  std::set<std::string> fixed;
  for(const auto& item:cfg["retarget"]["fixed_joints"]) {
   const std::string n=item.first.as<std::string>();auto it=std::find(names.begin(),names.end(),n);
   if(it==names.end() || !fixed.insert(n).second)fail("非法固定轴 "+n);
   const int i=std::distance(names.begin(),it);const double value=item.second.as<double>();
   if(!std::isfinite(value)||value<lower[i]||value>upper[i])fail("固定轴超出限位 "+n);
   solve_lower[i]=solve_upper[i]=value;
  }
  for(const auto* n:{"index_MCP_AA","middle_MCP_AA","ring_MCP_AA","pinky_MCP_AA"})
   if(!fixed.count(side+"_"+n))fail("缺少四指侧摆约束");
  optimizer=std::make_unique<AdaptiveRetarget>(cfg["retarget"],Eigen::Map<const V22>(solve_lower.data()),
      Eigen::Map<const V22>(solve_upper.data()),std::array<int,5>{2,7,11,15,20},std::array<int,5>{4,8,12,16,21},
      [this](const AdaptiveRetarget::Vector& q,AdaptiveRetarget::Kinematics& out){
       JointArray<22> angles{};std::copy_n(q.data(),22,angles.begin());Keypoints p;Jacobians j;kinematics(angles,p,&j);
       auto fill=[&](AdaptiveRetarget::TaskPoint& point,int index){point.position_cm=100*p[index];point.jacobian_cm=100*j[index];};
       fill(out.wrist,0);for(int finger=0;finger<5;++finger){fill(out.pip[finger],4*finger+2);fill(out.dip[finger],4*finger+3);fill(out.tip[finger],4*finger+4);}
      });
 }
 void validate_feedback(const JointArray<22>& q) const {
  for(int i=0;i<22;++i)if(!std::isfinite(q[i]))throw std::invalid_argument("Sharpa实际反馈关节非有限："+names[i]);
 }
 void validate(const JointArray<22>& q) const {
  for(int i=0;i<22;++i)if(!std::isfinite(q[i]) || q[i]<lower[i] || q[i]>upper[i])throw std::invalid_argument("Sharpa关节非有限或超出限位："+names[i]);
 }
 JointArray<22> projected(JointArray<22> q) const {for(int i=0;i<22;++i)q[i]=std::clamp(q[i],solve_lower[i],solve_upper[i]);return q;}
 void kinematics(const JointArray<22>& actual,Keypoints& p,Jacobians* jac,bool task_domain=true) {
  V22 q;for(int i=0;i<22;++i)q[semantic[i]]=actual[i];
  if(jac)pinocchio::computeJointJacobians(model,*data,q);else pinocchio::forwardKinematics(model,*data,q);
  pinocchio::updateFramePlacements(model,*data);
  const V3 wrist=data->oMf[frames[0]].translation();
  for(int i=0;i<21;++i) {
   p[i]=data->oMf[frames[i]].translation()-wrist;
   if(jac) {J6 j=J6::Zero();pinocchio::getFrameJacobian(model,*data,frames[i],pinocchio::LOCAL_WORLD_ALIGNED,j);for(int k=0;k<22;++k)(*jac)[i].col(k)=j.topRows<3>().col(semantic[k]);}
  }
  // 设备几何只在Sharpa任务域变换；原始角度/物理FK不改，正向与逆向使用同一任务域。
  if(task_domain)for(int i=1;i<21;++i){
   p[i]=(p[i]-wrist_offset).cwiseQuotient(task_axis_scale);
   if(jac)for(int row=0;row<3;++row)(*jac)[i].row(row)/=task_axis_scale[row];
  }
  // 共同21点约定：thumb CMC是腕与thumb MCP的辅助中点，非URDF基座CMC轴心。
  p[1]=.5*(p[0]+p[2]);
  if(jac)(*jac)[1]=.5*((*jac)[0]+(*jac)[2]);
  // 基座固定，腕Jacobian为零；URDF米坐标已与共同canonical一致。
 }
 JointArray<22> solve(const Points& input) {
  AdaptiveRetarget::ModelPoints points;
  for(int i=0;i<63;++i)if(!std::isfinite(input[i]))throw std::invalid_argument("Sharpa输入任务点非有限");
  for(int i=0;i<21;++i)points[i]=V3(input[3*i],input[3*i+1],input[3*i+2]);
  const V3 palm=points[9]-points[0];
  if(palm.norm()<=1e-6||palm.cross(points[5]-points[17]).norm()<=1e-8)
   throw std::invalid_argument("Sharpa输入手掌退化");
  const auto result=optimizer->solve(points);JointArray<22> candidate{};
  std::copy_n(result.data(),22,candidate.begin());
  // Sharpa求解器沿用Wuji float32输出；发送前仍严格投影本机命令边界。
  candidate=projected(candidate);validate(candidate);return candidate;
 }
};
Model::Model(std::filesystem::path root,bool left,std::filesystem::path config)
 :impl_(std::make_unique<Impl>(std::move(root),left,std::move(config))){}
Model::~Model()=default;Model::Model(Model&&) noexcept=default;Model& Model::operator=(Model&&) noexcept=default;
const std::array<std::string,22>& Model::joint_names()const{return impl_->names;}
const JointArray<22>& Model::lower_limits()const{return impl_->lower;}
const JointArray<22>& Model::upper_limits()const{return impl_->upper;}
const JointArray<22>& Model::effort_limits()const{return impl_->effort;}
JointArray<22> Model::solve(const Points& p){return impl_->solve(p);}
Points Model::hand_points(const JointArray<22>& q){impl_->validate_feedback(q);Keypoints p;impl_->kinematics(q,p,nullptr);Points result{};for(int i=0;i<21;++i)for(int k=0;k<3;++k)result[3*i+k]=p[i][k];return result;}
Points Model::physical_hand_points(const JointArray<22>& q){impl_->validate_feedback(q);Keypoints p;impl_->kinematics(q,p,nullptr,false);Points result{};for(int i=0;i<21;++i)for(int k=0;k<3;++k)result[3*i+k]=p[i][k];return result;}
std::array<double,63*22> Model::point_jacobian(const JointArray<22>& q){impl_->validate_feedback(q);Keypoints p;Jacobians j;impl_->kinematics(q,p,&j);std::array<double,63*22> out;for(int i=0;i<21;++i)for(int r=0;r<3;++r)for(int c=0;c<22;++c)out[(3*i+r)*22+c]=j[i](r,c);return out;}
std::array<double,21> Model::task_scales()const{
 auto scales=impl_->optimizer->task_scales();
 // 辅助CMC=p2/2，逆向除比例后仍须满足同一闭链中点约束。
 scales[1]=scales[2];return scales;
}
void Model::seed(const JointArray<22>& q){impl_->validate_feedback(q);const auto bounded=impl_->projected(q);impl_->optimizer->seed(Eigen::Map<const V22>(bounded.data()));}
void Model::reset(){impl_->optimizer->reset();}
void Model::set_initial_hint(const JointArray<22>* q){
 if(!q){impl_->optimizer->set_initial_hint(nullptr);return;}
 impl_->validate(*q);const auto bounded=impl_->projected(*q);
 AdaptiveRetarget::Vector hint=Eigen::Map<const V22>(bounded.data());impl_->optimizer->set_initial_hint(&hint);
}
std::unique_ptr<HandModel<22>> Model::create_solver(bool differentiation)const {
 auto solver=std::make_unique<Model>(impl_->root,impl_->left,impl_->config_path);
 if(differentiation)solver->impl_->optimizer->configure_differentiation();
 return solver;
}
} // namespace motorhand::native::sharpa
