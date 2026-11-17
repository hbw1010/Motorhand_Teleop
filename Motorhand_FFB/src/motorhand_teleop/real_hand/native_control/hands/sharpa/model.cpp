#include "model.hpp"
#include <Eigen/Core>
#include <nlopt.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <nlohmann/json.hpp>
#include <tinyxml2.h>
#include <algorithm>
#include <exception>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <type_traits>

namespace motorhand::native::sharpa {
namespace {
using V3=Eigen::Vector3d;
using V22=Eigen::Matrix<double,22,1>;
using J3=Eigen::Matrix<double,3,22>;
using J6=Eigen::Matrix<double,6,22>;
using Keypoints=std::array<V3,21>;
using Directions=std::array<V3,20>;
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
void nlcheck(nlopt_result r,const char* operation) {if(r<0)fail(std::string("NLopt配置失败 ")+operation);}
std::array<std::pair<int,int>,20> edges() {
 std::array<std::pair<int,int>,20> result;int k=0;
 for(int f=0;f<5;++f) {int s=1+4*f;result[k++]={0,s};for(int j=0;j<3;++j)result[k++]={s+j,s+j+1};}return result;
}
const auto bone_edges=edges();
}
struct Model::Impl {
 std::filesystem::path root;
 bool left{},differentiation{},has_hint{};
 pinocchio::Model model;
 std::unique_ptr<pinocchio::Data> data;
 std::array<std::string,22> names;
 JointArray<22> lower{},upper{},effort{},solve_lower{},solve_upper{},previous{},hint{};
 std::array<int,22> semantic{};
 std::array<pinocchio::FrameIndex,21> frames{};
 std::unique_ptr<std::remove_pointer_t<nlopt_opt>,decltype(&nlopt_destroy)> opt{nullptr,&nlopt_destroy};
 std::exception_ptr callback_error;
 Keypoints target{};
 Directions target_directions{};
 double scale{},w_dir{},w_pos{},w_temporal{},palm_length{};
 int maxeval{},diff_maxeval{};

 Impl(std::filesystem::path path,bool is_left):root(std::filesystem::canonical(path)),left(is_left) {
  const std::string side=left?"left":"right";
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
  pinocchio::urdf::buildModel(urdf.string(),model);
  if(model.nq!=22 || model.nv!=22)fail("Pinocchio维度不是22");
  for(int i=0;i<22;++i) {
   if(!model.existJointName(names[i]))fail("Pinocchio缺少关节");
   const auto& j=model.joints[model.getJointId(names[i])];
   if(j.nq()!=1 || j.nv()!=1 || j.idx_q()!=j.idx_v())fail("Pinocchio关节必须为标量");semantic[i]=j.idx_q();
  }
  auto frame=[&](const std::string& name) {if(!model.existFrame(name))fail("缺少任务frame "+name);return model.getFrameId(name);};
  frames[0]=frame(side+"_hand_C_MC");
  frames[1]=frame(side+"_thumb_CMC_VL");frames[2]=frame(side+"_thumb_MCP_VL");frames[3]=frame(side+"_thumb_DP");frames[4]=frame(side+"_thumb_fingertip");
  const std::array<std::string,4> fingers={"index","middle","ring","pinky"};
  const std::array<std::string,4> suffix={"MCP_VL","MP","DP","fingertip"};
  for(int f=0;f<4;++f)for(int k=0;k<4;++k)frames[5+4*f+k]=frame(side+"_"+fingers[f]+"_"+suffix[k]);
  data=std::make_unique<pinocchio::Data>(model);
  const auto config_path=root.parent_path().parent_path()/"config/retargeting/sharpa/model.json";
  std::ifstream in(config_path);if(!in)fail("无法读取重定向配置 "+config_path.string());
  const auto cfg=nlohmann::json::parse(in);
  if(cfg.at("schema_version").get<int>()!=1)fail("不支持的重定向配置版本");
  scale=cfg.at("position_scale");w_dir=cfg.at("direction_weight");w_pos=cfg.at("position_weight");w_temporal=cfg.at("temporal_weight");maxeval=cfg.at("max_evaluations");diff_maxeval=cfg.at("differentiation_max_evaluations");
  for(double v:{scale,w_dir,w_pos,w_temporal})if(!std::isfinite(v) || v<0)fail("非法重定向权重");
  if(scale<=0 || w_dir<=0 || maxeval<=0 || diff_maxeval<=0)fail("非法重定向比例或迭代次数");
  solve_lower=lower;solve_upper=upper;
  std::set<std::string> fixed;
  const double fixed_value=cfg.at("fixed_joint_value_rad");
  if(fixed_value!=0)fail("不可观测轴必须固定零");
  for(const auto& item:cfg.at("fixed_joint_suffixes")) {
   const std::string n=side+"_"+item.get<std::string>();auto it=std::find(names.begin(),names.end(),n);
   if(it==names.end() || !fixed.insert(n).second)fail("非法固定轴 "+n);
   const int i=std::distance(names.begin(),it);solve_lower[i]=solve_upper[i]=0;
  }
  for(const auto* n:{"thumb_MCP_AA","index_MCP_AA","middle_MCP_AA","ring_MCP_AA","pinky_CMC","pinky_MCP_AA"})if(!fixed.count(side+"_"+n))fail("缺少不可观测轴约束");
  Keypoints p;kinematics(previous,p,nullptr);palm_length=p[9].norm();if(palm_length<=1e-6)fail("退化掌长");
  opt.reset(nlopt_create(NLOPT_LD_SLSQP,22));if(!opt)fail("无法创建NLopt");
  nlcheck(nlopt_set_lower_bounds(opt.get(),solve_lower.data()),"lower");nlcheck(nlopt_set_upper_bounds(opt.get(),solve_upper.data()),"upper");
  nlcheck(nlopt_set_maxeval(opt.get(),maxeval),"maxeval");nlcheck(nlopt_set_ftol_abs(opt.get(),1e-10),"ftol");nlcheck(nlopt_set_xtol_rel(opt.get(),1e-8),"xtol");nlcheck(nlopt_set_min_objective(opt.get(),&objective,this),"objective");
 }
 void validate_feedback(const JointArray<22>& q) const {
  for(int i=0;i<22;++i)if(!std::isfinite(q[i]))throw std::invalid_argument("Sharpa实际反馈关节非有限："+names[i]);
 }
 void validate(const JointArray<22>& q) const {
  for(int i=0;i<22;++i)if(!std::isfinite(q[i]) || q[i]<lower[i] || q[i]>upper[i])throw std::invalid_argument("Sharpa关节非有限或超出限位："+names[i]);
 }
 JointArray<22> projected(JointArray<22> q) const {for(int i=0;i<22;++i)q[i]=std::clamp(q[i],solve_lower[i],solve_upper[i]);return q;}
 void kinematics(const JointArray<22>& actual,Keypoints& p,Jacobians* jac) {
  V22 q;for(int i=0;i<22;++i)q[semantic[i]]=actual[i];
  if(jac)pinocchio::computeJointJacobians(model,*data,q);else pinocchio::forwardKinematics(model,*data,q);
  pinocchio::updateFramePlacements(model,*data);
  const V3 wrist=data->oMf[frames[0]].translation();
  for(int i=0;i<21;++i) {
   p[i]=data->oMf[frames[i]].translation()-wrist;
   if(jac) {J6 j=J6::Zero();pinocchio::getFrameJacobian(model,*data,frames[i],pinocchio::LOCAL_WORLD_ALIGNED,j);for(int k=0;k<22;++k)(*jac)[i].col(k)=j.topRows<3>().col(semantic[k]);}
  }
  // 共同21点约定：thumb CMC是腕与thumb MCP的辅助中点，非URDF基座CMC轴心。
  p[1]=.5*(p[0]+p[2]);
  if(jac)(*jac)[1]=.5*((*jac)[0]+(*jac)[2]);
  // 基座固定，腕Jacobian为零；URDF米坐标已与共同canonical一致。
 }
 static double objective(unsigned n,const double* x,double* gradient,void* context) noexcept {
  auto& s=*static_cast<Impl*>(context);
  try {if(n!=22)fail("NLopt维度错误");JointArray<22> q;std::copy(x,x+22,q.begin());V22 g=V22::Zero();const double cost=s.loss(q,g);if(!std::isfinite(cost)||!g.allFinite())fail("重定向目标非有限");if(gradient)std::copy(g.data(),g.data()+22,gradient);return cost;}
  catch(...) {s.callback_error=std::current_exception();nlopt_force_stop(s.opt.get());return 0;}
 }
 double loss(const JointArray<22>& q,V22& g) {
  Keypoints p;Jacobians jac;kinematics(q,p,&jac);double cost=0;
  for(int i=0;i<20;++i) {
   const auto [a,b]=bone_edges[i];const V3 v=p[b]-p[a];const double length=v.norm();if(length<=1e-8)fail("模型骨段退化");const V3 unit=v/length;
   const V3 error=unit-target_directions[i];const J3 derivative=(Eigen::Matrix3d::Identity()-unit*unit.transpose())*(jac[b]-jac[a])/length;
   // 掌部固定骨段不提供关节信息，其常量误差不影响优化。
   cost+=w_dir*error.squaredNorm();g.noalias()+=2*w_dir*derivative.transpose()*error;
  }
  for(int i:{4,8,12,16,20}) {const V3 e=(p[i]-target[i])/palm_length;cost+=w_pos*e.squaredNorm();g.noalias()+=2*w_pos/palm_length*jac[i].transpose()*e;}
  if(!differentiation)for(int i=0;i<22;++i) {const double d=q[i]-previous[i];cost+=w_temporal*d*d;g[i]+=2*w_temporal*d;}
  return cost;
 }
 JointArray<22> solve(const Points& input) {
  // 完整检查先于任何热启动/目标修改，非法帧不改变后续结果。
  Keypoints checked;Directions directions;
  for(int i=0;i<63;++i)if(!std::isfinite(input[i]))throw std::invalid_argument("Sharpa输入任务点非有限");
  const V3 wrist(input[0],input[1],input[2]);
  for(int i=0;i<21;++i)checked[i]=scale*(V3(input[3*i],input[3*i+1],input[3*i+2])-wrist);
  for(int i=0;i<20;++i) {const auto [a,b]=bone_edges[i];const V3 v=checked[b]-checked[a];if(v.norm()<=1e-6)throw std::invalid_argument("Sharpa输入骨段退化");directions[i]=v.normalized();}
  if(checked[9].norm()<=1e-6 || checked[9].cross(checked[5]-checked[17]).norm()<=1e-8)throw std::invalid_argument("Sharpa输入手掌退化");
  target=checked;target_directions=directions;callback_error=nullptr;
  JointArray<22> candidate=projected(has_hint?hint:(differentiation?JointArray<22>{}:previous));
  double value=0;const auto result=nlopt_optimize(opt.get(),candidate.data(),&value);
  if(callback_error)std::rethrow_exception(callback_error);
  if(result<0 && result!=NLOPT_ROUNDOFF_LIMITED)fail("重定向求解失败，状态="+std::to_string(result));
  candidate=projected(candidate);validate(candidate);if(!std::isfinite(value))fail("重定向结果非有限");
  if(!differentiation)previous=candidate;
  return candidate;
 }
};
Model::Model(std::filesystem::path root,bool left):impl_(std::make_unique<Impl>(std::move(root),left)){}
Model::~Model()=default;Model::Model(Model&&) noexcept=default;Model& Model::operator=(Model&&) noexcept=default;
const std::array<std::string,22>& Model::joint_names()const{return impl_->names;}
const JointArray<22>& Model::lower_limits()const{return impl_->lower;}
const JointArray<22>& Model::upper_limits()const{return impl_->upper;}
const JointArray<22>& Model::effort_limits()const{return impl_->effort;}
JointArray<22> Model::solve(const Points& p){return impl_->solve(p);}
Points Model::hand_points(const JointArray<22>& q){impl_->validate_feedback(q);Keypoints p;impl_->kinematics(q,p,nullptr);Points result{};for(int i=0;i<21;++i)for(int k=0;k<3;++k)result[3*i+k]=p[i][k];return result;}
std::array<double,63*22> Model::point_jacobian(const JointArray<22>& q){impl_->validate_feedback(q);Keypoints p;Jacobians j;impl_->kinematics(q,p,&j);std::array<double,63*22> out;for(int i=0;i<21;++i)for(int r=0;r<3;++r)for(int c=0;c<22;++c)out[(3*i+r)*22+c]=j[i](r,c);return out;}
std::array<double,21> Model::task_scales()const{std::array<double,21> s;s.fill(impl_->scale);return s;}
void Model::seed(const JointArray<22>& q){impl_->validate_feedback(q);impl_->previous=impl_->projected(q);}
void Model::reset(){impl_->previous={};}
void Model::set_initial_hint(const JointArray<22>* q){if(q){impl_->validate(*q);impl_->hint=impl_->projected(*q);}impl_->has_hint=q!=nullptr;}
std::unique_ptr<HandModel<22>> Model::create_solver(bool differentiation)const {
 auto s=std::make_unique<Model>(impl_->root,impl_->left);s->impl_->differentiation=differentiation;
 if(differentiation) {nlcheck(nlopt_set_maxeval(s->impl_->opt.get(),impl_->diff_maxeval),"微分maxeval");nlcheck(nlopt_set_ftol_abs(s->impl_->opt.get(),1e-13),"微分ftol");}
 else s->impl_->previous=impl_->previous;
 return s;
}
} // namespace motorhand::native::sharpa
