#include "sharpa_device.hpp"
#include <mujoco/mujoco.h>
#include <tinyxml2.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

namespace motorhand::native::simulation {
namespace {
using Clock=std::chrono::steady_clock;
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
[[noreturn]] void fail(const std::string& s){throw std::runtime_error("Sharpa仿真："+s);}
std::string make_scene(const std::filesystem::path& root,bool left,bool object){
 tinyxml2::XMLDocument xml;const std::string side=left?"left":"right";
 if(xml.LoadFile((root/(side+".xml")).string().c_str())!=tinyxml2::XML_SUCCESS)fail("无法读取MJCF");
 auto* scene=xml.FirstChildElement("mujoco");if(!scene)fail("无效MJCF");
 auto* assets=scene->FirstChildElement("asset");if(!assets)fail("缺少mesh资产");
 for(auto* mesh=assets->FirstChildElement("mesh");mesh;mesh=mesh->NextSiblingElement("mesh")){
  const char* file=mesh->Attribute("file");if(!file)fail("mesh缺少file");
  const auto absolute=std::filesystem::canonical(root/file).string();mesh->SetAttribute("file",absolute.c_str());
 }
 auto* compiler=scene->FirstChildElement("compiler");if(compiler)compiler->SetAttribute("meshdir","");
 auto* option=scene->FirstChildElement("option");if(!option){option=xml.NewElement("option");scene->InsertEndChild(option);}
 option->SetAttribute("timestep",.002);
 if(object){
  tinyxml2::XMLElement* palm=nullptr;
  auto find=[&](auto&& self,tinyxml2::XMLElement* e)->void{
   if(!e||palm)return;
   if(std::string(e->Name())=="body" && e->Attribute("name") && std::string(e->Attribute("name"))==side+"_hand_C_MC"){palm=e;return;}
   for(auto* c=e->FirstChildElement();c;c=c->NextSiblingElement())self(self,c);
  };find(find,scene->FirstChildElement("worldbody"));if(!palm)fail("缺少hand_C_MC局部坐标");
  // 无关节的子体与基座锁定。物体表面位于屈曲侧，伸直时留有间隙。
  auto* body=xml.NewElement("body");body->SetAttribute("name","teleop_locked_object");body->SetAttribute("pos","0.065 0 0.14");
  auto* geom=xml.NewElement("geom");geom->SetAttribute("name","teleop_locked_object_geom");geom->SetAttribute("type","sphere");geom->SetAttribute("size","0.035");geom->SetAttribute("rgba","0.8 0.32 0.16 1");geom->SetAttribute("friction","0.6 0.01 0.001");geom->SetAttribute("contype",1);geom->SetAttribute("conaffinity",1);body->InsertEndChild(geom);palm->InsertEndChild(body);
 }
 tinyxml2::XMLPrinter printer;xml.Print(&printer);return printer.CStr();
}
}
struct SharpaDevice::Impl {
 std::filesystem::path root;
 HandDescriptor<22> descriptor;
 bool object{},connected{},active{},homed{},stop{},closed{},actuators_ready{};
 std::uint64_t deadline{};
 mutable std::mutex mutex;
 std::condition_variable cv;
 std::thread worker;
 std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model{nullptr,&mj_deleteModel};
 std::unique_ptr<mjData,decltype(&mj_deleteData)> data{nullptr,&mj_deleteData};
 std::array<int,22> qpos{},dof{},actuator{};
 std::array<std::array<mjtNum,mjNGAIN>,22> gains{};
 std::array<std::array<mjtNum,mjNBIAS>,22> biases{};
 std::vector<mjtNum> jacp,jacr;
 int object_geom=-1;
 std::vector<unsigned char> hand_body;
 HandState<22> state;
 SharpaSnapshot snapshot;
 Impl(std::filesystem::path path,HandDescriptor<22> d,bool with_object):root(std::filesystem::canonical(path)),descriptor(std::move(d)),object(with_object){
  const bool left=descriptor.identity.left;const std::string prefix=left?"left_":"right_";std::set<std::string> names;
  for(int i=0;i<22;++i){
   if(descriptor.joint_names[i].rfind(prefix,0)!=0 || !names.insert(descriptor.joint_names[i]).second || !std::isfinite(descriptor.lower_rad[i]) || !std::isfinite(descriptor.upper_rad[i]) || descriptor.lower_rad[i]>=descriptor.upper_rad[i] || descriptor.lower_rad[i]>0 || descriptor.upper_rad[i]<0)fail("无效22轴描述或零位");
  }
  descriptor.identity.model="sharpa";descriptor.identity.serial=left?"SIM-SHARPA-LEFT":"SIM-SHARPA-RIGHT";descriptor.identity.firmware="mujoco-4.0.0";
  descriptor.online_source=HandStatusSource::complete_state_packet;descriptor.enabled_source=HandStatusSource::device_report;descriptor.load_unit=HandLoadUnit::newton_metre;descriptor.load_source=sharpa_contact_load_source;
  state.identity=descriptor.identity;state.online_source=descriptor.online_source;state.enabled_source=descriptor.enabled_source;state.load_unit=descriptor.load_unit;state.load_source=descriptor.load_source;state.has_velocity=state.has_load=true;
 }
 void ready()const{if(!connected||state.fault||closed)fail("实例未连接或已故障/关闭");}
 void actuators_enabled(bool enabled){
  for(int i=0;i<22;++i){
   auto* gain=model->actuator_gainprm+mjNGAIN*actuator[i];auto* bias=model->actuator_biasprm+mjNBIAS*actuator[i];
   for(int k=0;k<mjNGAIN;++k)gain[k]=enabled?gains[i][k]:0;
   for(int k=0;k<mjNBIAS;++k)bias[k]=enabled?biases[i][k]:0;
  }
 }
 void disable_locked(){active=false;homed=false;deadline=0;if(model&&actuators_ready)actuators_enabled(false);state.enabled.fill(false);}
 void write_target(const JointArray<22>& q){for(int i=0;i<22;++i)data->ctrl[actuator[i]]=q[i];snapshot.target_rad=q;}
 void publish(){
  state.connected=connected;state.online.fill(connected);state.enabled.fill(active);state.load.fill(0);snapshot.contact_count=0;
  if(object_geom>=0){
   for(int c=0;c<data->ncon;++c){
    const auto& contact=data->contact[c];int hand_geom=-1;double sign=0;
    if(contact.geom[0]==object_geom){hand_geom=contact.geom[1];sign=1;}
    else if(contact.geom[1]==object_geom){hand_geom=contact.geom[0];sign=-1;}
    if(hand_geom<0 || !hand_body[model->geom_bodyid[hand_geom]] || contact.efc_address<0)continue;
    ++snapshot.contact_count;
    mjtNum local[6]{};mj_contactForce(model.get(),data.get(),c,local);
    mjtNum force[3]{},torque[3]{};
    // contact.frame的行是接触坐标轴；法向从geom[0]指向geom[1]。
    // mj_contactForce给geom[1]的作用力；选择手所在一侧并转到世界坐标。
    for(int r=0;r<3;++r)for(int k=0;k<3;++k){force[r]+=sign*contact.frame[3*k+r]*local[k];torque[r]+=sign*contact.frame[3*k+r]*local[k+3];}
    mj_jac(model.get(),data.get(),jacp.data(),jacr.data(),contact.pos,model->geom_bodyid[hand_geom]);
    for(int i=0;i<22;++i){double environment=0;for(int r=0;r<3;++r)environment+=jacp[r*model->nv+dof[i]]*force[r]+jacr[r*model->nv+dof[i]]*torque[r];state.load[i]-=environment;}
   }
  }
  for(int i=0;i<22;++i){
   state.position_rad[i]=data->qpos[qpos[i]];state.velocity_rad_s[i]=data->qvel[dof[i]];
   if(!std::isfinite(state.position_rad[i])||!std::isfinite(state.velocity_rad_s[i])||!std::isfinite(state.load[i]))fail("物理状态非有限");
  }
  snapshot.position_rad=state.position_rad;snapshot.contact_load_nm=state.load;snapshot.simulation_time=data->time;snapshot.qpos.assign(data->qpos,data->qpos+model->nq);
  state.received_ns=now_ns();state.device_time=static_cast<std::uint64_t>(data->time*1e9);++state.sequence;
 }
 void run()noexcept{
  auto next=Clock::now();
  try{
   std::unique_lock<std::mutex> lock(mutex);
   while(!stop){
    if(active&&now_ns()>=deadline)disable_locked();
    mj_step(model.get(),data.get());
    // mj_step积分后的qpos与接触约束重新同步，负载/雅可比/缓存同一物理状态。
    mj_forward(model.get(),data.get());publish();
    next+=std::chrono::milliseconds(2);
    if(next<Clock::now()-std::chrono::milliseconds(20))next=Clock::now();
    cv.wait_until(lock,next,[&]{return stop;});
   }
  }catch(...){std::lock_guard<std::mutex> lock(mutex);state.fault=true;disable_locked();stop=true;cv.notify_all();}
 }
};
SharpaDevice::SharpaDevice(std::filesystem::path root,HandDescriptor<22> d,bool object):impl_(std::make_unique<Impl>(std::move(root),std::move(d),object)){}
SharpaDevice::~SharpaDevice(){close();}
const HandDescriptor<22>& SharpaDevice::descriptor()const{return impl_->descriptor;}
void SharpaDevice::connect(){
 auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);if(s.connected||s.closed)fail("不能重复连接或重用已关闭实例");
 if(std::string(mj_versionString())!="4.0.0")fail("需要固定版本MuJoCo 4.0.0");
 s.snapshot.scene_xml=make_scene(s.root,s.descriptor.identity.left,s.object);char error[2048]{};
 std::unique_ptr<mjSpec,decltype(&mj_deleteSpec)> spec(mj_parseXMLString(s.snapshot.scene_xml.c_str(),nullptr,error,sizeof(error)),&mj_deleteSpec);if(!spec)fail(std::string("MJCF解析失败：")+error);
 s.model.reset(mj_compile(spec.get(),nullptr));if(!s.model)fail(std::string("MJCF编译失败：")+mjs_getError(spec.get()));
 if(s.model->nq!=22||s.model->nv!=22||s.model->nu!=22 || std::abs(s.model->opt.timestep-.002)>1e-12)fail("模型必须为22轴、22执行器、0.002秒步长");
 s.data.reset(mj_makeData(s.model.get()));if(!s.data)fail("无法创建mjData");
 std::set<int> distinct;const std::string side=s.descriptor.identity.left?"left":"right";
 for(int i=0;i<22;++i){
  const auto& name=s.descriptor.joint_names[i];const int joint=mj_name2id(s.model.get(),mjOBJ_JOINT,name.c_str());if(joint<0||s.model->jnt_type[joint]!=mjJNT_HINGE)fail("缺少语义转动关节 "+name);
  s.qpos[i]=s.model->jnt_qposadr[joint];s.dof[i]=s.model->jnt_dofadr[joint];int selected=-1;
  for(int a=0;a<s.model->nu;++a)if(s.model->actuator_trntype[a]==mjTRN_JOINT&&s.model->actuator_trnid[2*a]==joint){if(selected>=0)fail("关节执行器重复");selected=a;}
  if(selected<0||!distinct.insert(selected).second)fail("缺少命名执行器 "+name);s.actuator[i]=selected;
  std::copy_n(s.model->actuator_gainprm+mjNGAIN*selected,mjNGAIN,s.gains[i].begin());std::copy_n(s.model->actuator_biasprm+mjNBIAS*selected,mjNBIAS,s.biases[i].begin());
 }
 s.actuators_ready=true;
 s.object_geom=mj_name2id(s.model.get(),mjOBJ_GEOM,"teleop_locked_object_geom");
 const int palm=mj_name2id(s.model.get(),mjOBJ_BODY,(side+"_hand_C_MC").c_str());if(palm<0)fail("缺少手掌body");
 s.hand_body.assign(s.model->nbody,0);
 for(int b=1;b<s.model->nbody;++b){int ancestor=b;while(ancestor>0&&ancestor!=palm)ancestor=s.model->body_parentid[ancestor];s.hand_body[b]=ancestor==palm;}
 if(s.object_geom>=0)s.hand_body[s.model->geom_bodyid[s.object_geom]]=0;
 s.jacp.resize(3*s.model->nv);s.jacr.resize(3*s.model->nv);s.actuators_enabled(false);mj_forward(s.model.get(),s.data.get());s.connected=true;s.publish();s.worker=std::thread([&s]{s.run();});
}
HandState<22> SharpaDevice::read(){std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->state;}
SharpaSnapshot SharpaDevice::snapshot()const{std::lock_guard<std::mutex> lock(impl_->mutex);return impl_->snapshot;}
void SharpaDevice::enable(){auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.ready();if(s.active)fail("重复使能");JointArray<22> q;for(int i=0;i<22;++i)q[i]=std::clamp(s.data->qpos[s.qpos[i]],s.descriptor.lower_rad[i],s.descriptor.upper_rad[i]);s.write_target(q);s.actuators_enabled(true);s.active=true;s.homed=false;s.deadline=now_ns()+hand_timeout_ns;s.state.enabled.fill(true);}
void SharpaDevice::home(const std::atomic<bool>* cancel){
 auto& s=*impl_;const auto end=Clock::now()+std::chrono::seconds(6);std::unique_lock<std::mutex> lock(s.mutex);s.ready();if(!s.active)fail("回零前必须使能");s.write_target(JointArray<22>{});
 while(true){
  if(cancel&&cancel->load()){s.disable_locked();fail("回零已取消");}s.ready();if(!s.active)fail("回零期间已失能");
  bool arrived=true;for(int i=0;i<22;++i)arrived&=std::abs(s.state.position_rad[i])<.04&&std::abs(s.state.velocity_rad_s[i])<.15;
  s.deadline=now_ns()+hand_timeout_ns;
  if(arrived){s.homed=true;return;}if(Clock::now()>=end){s.disable_locked();fail("物理回零超时");}
  s.cv.wait_for(lock,std::chrono::milliseconds(5));
 }
}
void SharpaDevice::send(const JointArray<22>& q,std::uint64_t source){
 auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.ready();if(!s.active||!s.homed)fail("发送前必须使能并回零");
 for(int i=0;i<22;++i)if(!std::isfinite(q[i])||q[i]<s.descriptor.lower_rad[i]||q[i]>s.descriptor.upper_rad[i])fail("目标非有限或超出限位");
 const auto now=now_ns();if(source<=now || !s.state.received_ns || s.state.received_ns>now || now-s.state.received_ns>=hand_timeout_ns)fail("目标原始期限或物理反馈已过期");
 s.deadline=std::min({source,now+hand_timeout_ns,s.state.received_ns+hand_timeout_ns});s.write_target(q);
 if(now_ns()>=s.deadline){s.disable_locked();fail("目标写入跨越原始期限");}
}
void SharpaDevice::disable(){auto& s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);s.disable_locked();}
void SharpaDevice::close(){
 if(!impl_)return;auto& s=*impl_;
 {std::lock_guard<std::mutex> lock(s.mutex);s.stop=true;s.closed=true;s.connected=false;s.state.connected=false;s.state.online.fill(false);s.disable_locked();s.cv.notify_all();}
 if(s.worker.joinable())s.worker.join();
}
} // namespace motorhand::native::simulation
