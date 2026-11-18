#include "motorhand_teleop/real_hand/native_control/simulation/sharpa_device.hpp"
#include "motorhand_teleop/real_hand/native_control/simulation/fake_glove.hpp"
#include "motorhand_teleop/real_hand/native_control/common/driver_impedance.hpp"
#include "motorhand_teleop/real_hand/native_control/hands/sharpa/model.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
#include <tinyxml2.h>
using namespace motorhand::native;
using namespace motorhand::native::simulation;
using Clock=std::chrono::steady_clock;
std::uint64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
void check(bool v,const char* why){if(!v)throw std::runtime_error(why);}
template<class F>void throws(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"应拒绝非法操作");}
HandDescriptor<22> descriptor(const std::filesystem::path& assets,bool left){
 sharpa::Model model(assets,left);HandDescriptor<22>d;d.identity.left=left;d.identity.model="sharpa";
 d.joint_names=model.joint_names();d.lower_rad=model.lower_limits();d.upper_rad=model.upper_limits();return d;
}
void run_for(SharpaDevice& d,const JointArray<22>& q,int milliseconds){
 const auto end=Clock::now()+std::chrono::milliseconds(milliseconds);
 while(Clock::now()<end){d.send(q,now_ns()+hand_timeout_ns);std::this_thread::sleep_for(std::chrono::milliseconds(10));}
}
std::array<EncosMotorCalibration, 12> calibrated_glove(double torque_limit_nm) {
 std::array<EncosMotorCalibration, 12> calibration{};
 for (std::size_t i = 0; i < calibration.size(); ++i) {
  auto& axis = calibration[i];
  axis.raw_motor_id = i + 1;
  axis.sign = axis.scale = 1;
  axis.model_min_rad = -1;
  axis.model_max_rad = 1;
  axis.raw_torque_limit_nm = torque_limit_nm;
  axis.has_torque_limit = true;
 }
 return calibration;
}
void glove_respects_impedance_budget() {
 const auto calibration = calibrated_glove(4);
 FakeGlove glove(calibration);
 glove.connect();
 glove.home();
 glove.enable_force();
 DriverImpedanceCommand command;
 command.target_rad[5] = .1;
 command.kp[5] = 40;
 command.blocked_direction[5] = -1;
 const auto applied = glove.write_impedance(command, now_ns() + hand_timeout_ns);
 check(applied.nominal_nm[5] > 3.3 && applied.nominal_nm[5] <= 3.4,
       "仿真手套应在提交时重新限制阻抗名义力矩");
 check(std::abs(applied.nominal_nm[5] - applied.command.kp[5] * .1) < 1e-10,
       "仿真名义力矩应对应下发的阻抗增益");

 DriverImpedanceConfig config;
 config.torque_margin = .95;
 config.stiffness_min = config.stiffness_max = 40;
 config.surface_rate = 100;
 config.ramp_s = 0;
 CommonDriverImpedancePlanner<22> planner(config, .3, calibration);
 GlovePose pose;
 for (std::size_t i = 0; i < calibration.size(); ++i)
  pose.motor_jacobian[12 * i + i] = 1;
 RetargetJacobian<22> jacobian{};
 jacobian[5] = 1;
 HandState<22> hand;
 JointArray<22> target{};
 target[0] = .12;
 FeedbackResult<22> contact;
 contact.active = contact.contact[0] = contact.impedance_contact[0] = true;
 contact.contact_direction[0] = 1;
 const auto frame = glove.read();
 const auto now = now_ns();
 planner.reset(now);
 const auto plan = planner.step(target, hand, frame, pose, jacobian, contact, now);
 check(plan.valid && std::abs(plan.nominal_nm[5]) > 3.7, "规划器应使用95%预算");
 const auto submitted = glove.write_impedance(plan.command, now + hand_timeout_ns);
 const double torque = std::abs(submitted.nominal_nm[5]);
 check(torque > 3.7 && torque <= 3.8, "仿真提交应使用规划器的95%预算");
}
void contact_surface_builds_and_releases_force() {
 DriverImpedanceConfig config;
 config.torque_margin = .95;
 config.joint_gain.fill(2.7);
 config.surface_rate = .25;
 config.surface_acceleration = 5;
 config.ramp_s = 0;
 CommonDriverImpedancePlanner<22> planner(config, 1, calibrated_glove(10));
 GlovePose pose;
 for (int i = 0; i < 12; ++i)
  pose.motor_jacobian[12 * i + i] = 1;
 RetargetJacobian<22> jacobian{};
 jacobian[0] = 1;
 HandState<22> hand;
 GloveFrame frame;
 JointArray<22> target{};
 target[0] = .1;
 FeedbackResult<22> contact;
 contact.active = contact.contact[0] = contact.impedance_contact[0] = true;
 contact.contact_direction[0] = 1;
 std::uint64_t timestamp = 1'000'000'000;
 planner.reset(timestamp);
 auto step = [&] {
  frame.oldest_ns = timestamp;
  frame.received_ns.fill(timestamp);
  const auto result = planner.step(target, hand, frame, pose, jacobian, contact, timestamp);
  timestamp += 20'000'000;
  return result;
 };
 double previous_position = 0;
 double previous_velocity = 0;
 CommonDriverImpedancePlan<22> plan;
 for (int sample = 0; sample < 40; ++sample) {
  plan = step();
  const double velocity = (plan.proxy_surface_rad[0] - previous_position) / .02;
  check(plan.valid && std::abs(velocity) <= .25 + 1e-9, "阻挡目标必须限速");
  check(std::abs(velocity - previous_velocity) <= .1 + 1e-9, "阻挡目标必须按加速度渐入");
  previous_position = plan.proxy_surface_rad[0];
  previous_velocity = velocity;
 }
 check(std::abs(previous_position + .1 / 1.01) < 1e-3, "持续夹持应到达预估阻挡位置");
 check(std::abs(plan.nominal_nm[0]) > 9 && std::abs(plan.nominal_nm[0]) <= 9.5,
       "渐入后保持高刚度反馈并遵守名义预算");
 target[0] = .06;
 const auto repeated = planner.step(target, hand, frame, pose, jacobian, contact, frame.oldest_ns);
 check(repeated.proxy_surface_rad[0] == previous_position, "重复采样不能推进阻挡目标");
 plan = step();
 const double next_velocity = (plan.proxy_surface_rad[0] - previous_position) / .02;
 check(std::abs(next_velocity - previous_velocity) <= .1 + 1e-9 && plan.nominal_nm[0] <= 0,
       "目标回移时应平滑反向且保持原有抵抗方向");
 contact.impedance_contact[0] = false;
 plan = step();
 check(plan.command.kp[0] == 0 && plan.nominal_nm[0] == 0 && plan.proxy_surface_rad[0] == 0,
       "松手后必须立即清除阻挡目标和PD");
 contact.impedance_contact[0] = true;
 plan = step();
 check(std::abs(plan.proxy_surface_rad[0] + .002) < 1e-9, "再次接触必须从零速度建立阻碍");
}
int main(int argc,char**argv){try{
 glove_respects_impedance_budget();
 contact_surface_builds_and_releases_force();
 auto repo=argc>1?std::filesystem::path(argv[1]):std::filesystem::current_path().parent_path();auto assets=repo/"Motorhand_FFB/assets/sharpa-hand";

 {
  const auto invalid_root=repo/".cache/sharpa-simulation-test/invalid-model";std::filesystem::create_directories(invalid_root);
  tinyxml2::XMLDocument xml;check(xml.LoadFile((assets/"right.xml").string().c_str())==tinyxml2::XML_SUCCESS,"读取异常资产fixture");auto* scene=xml.FirstChildElement("mujoco");
  for(auto* mesh=scene->FirstChildElement("asset")->FirstChildElement("mesh");mesh;mesh=mesh->NextSiblingElement("mesh"))mesh->SetAttribute("file",std::filesystem::canonical(assets/mesh->Attribute("file")).string().c_str());
  scene->DeleteChild(scene->FirstChildElement("actuator"));check(xml.SaveFile((invalid_root/"right.xml").string().c_str())==tinyxml2::XML_SUCCESS,"写入异常模型fixture");
  SharpaDevice invalid(invalid_root,descriptor(assets,false),false);throws([&]{invalid.connect();});invalid.close();std::filesystem::remove_all(invalid_root);
 }
 for(bool left:{false,true}){
  SharpaDevice d(assets,descriptor(assets,left),false);d.connect();
  check(d.descriptor().identity.serial==(left?"SIM-SHARPA-LEFT":"SIM-SHARPA-RIGHT"),"仿真身份");
  auto initial=d.read();check(initial.received_ns>0&&initial.has_velocity&&initial.has_load,"原始缓存反馈完整");
  check(initial.load_source==sharpa_contact_load_source&&initial.load_unit==HandLoadUnit::newton_metre,"接触负载来源与单位");

  bool saw_same_sequence=false;
  for(int i=0;i<1000&&!saw_same_sequence;++i){auto a=d.read(),b=d.read();if(a.sequence==b.sequence){saw_same_sequence=true;check(a.received_ns==b.received_ns,"read不刷新物理反馈时间");}}
  check(saw_same_sequence,"能重复读取同一物理缓存");
  d.enable();std::atomic<bool> cancelled{true};throws([&]{d.home(&cancelled);});check(!d.read().enabled[0],"取消物理回零会失能");
  d.enable();d.home();auto before=d.snapshot();JointArray<22> q{};q[5]=.65;q[7]=.7;q[8]=.5;
  d.send(q,now_ns()+hand_timeout_ns);auto immediate=d.snapshot();
  check(immediate.target_rad==q,"目标进入ctrl缓存");
  check(std::abs(immediate.position_rad[5]-q[5])>.3,"send没有覆盖qpos，保留真实滞后");
  run_for(d,q,650);auto later=d.snapshot();
  check(later.simulation_time>before.simulation_time+.3,"独立500Hz物理推进");
  check(std::abs(later.position_rad[5]-q[5])<.2,"无物体时位置执行器跟随");
  double lower_overshoot=0,upper_overshoot=0;
  for(int i=0;i<22;++i){lower_overshoot=std::max(lower_overshoot,d.descriptor().lower_rad[i]-later.position_rad[i]);upper_overshoot=std::max(upper_overshoot,later.position_rad[i]-d.descriptor().upper_rad[i]);}
  std::cout<<(left?"左":"右")<<"手软限位下/上超调="<<lower_overshoot<<"/"<<upper_overshoot<<" rad\n";

  sharpa::Model feedback_model(assets,left);auto actual_points=feedback_model.hand_points(later.position_rad);feedback_model.seed(later.position_rad);
  const auto task_scale=feedback_model.task_scales();for(int i=0;i<63;++i)actual_points[i]/=task_scale[i/3];const auto legal_target=feedback_model.solve(actual_points);
  for(int i=0;i<22;++i)check(legal_target[i]>=feedback_model.lower_limits()[i]&&legal_target[i]<=feedback_model.upper_limits()[i],"真实软限位反馈经模型FK/seed后命令仍合法");
  for(double load:later.contact_load_nm)check(std::abs(load)<1e-12,"无物体不混入重力驱动或关节内力");
  auto invalid=q;invalid[1]=std::numeric_limits<double>::quiet_NaN();throws([&]{d.send(invalid,now_ns()+hand_timeout_ns);});
  invalid=q;invalid[0]=3;throws([&]{d.send(invalid,now_ns()+hand_timeout_ns);});
  throws([&]{d.send(q,now_ns()-1);});
  check(d.snapshot().target_rad==q,"非法发送不污染目标");
  d.disable();check(!d.read().enabled[0],"disable撤销执行器使能");throws([&]{d.send(q,now_ns()+hand_timeout_ns);});
  d.close();auto stopped=d.snapshot();std::this_thread::sleep_for(std::chrono::milliseconds(15));
  check(d.snapshot().simulation_time==stopped.simulation_time,"close停止仿真线程");check(!d.read().connected,"close连接状态");throws([&]{d.enable();});throws([&]{d.connect();});d.disable();
  auto stopped_state=d.read();std::this_thread::sleep_for(std::chrono::milliseconds(5));check(d.read().received_ns==stopped_state.received_ns,"close后read不伪造新反馈时间");
 }

 auto reordered=descriptor(assets,true);
 std::reverse(reordered.joint_names.begin(),reordered.joint_names.end());std::reverse(reordered.lower_rad.begin(),reordered.lower_rad.end());std::reverse(reordered.upper_rad.begin(),reordered.upper_rad.end());
 SharpaDevice mapped(assets,reordered,false);mapped.connect();mapped.enable();mapped.home();
 JointArray<22> named_target{};const auto found=std::find(reordered.joint_names.begin(),reordered.joint_names.end(),"left_index_MCP_FE");const int named_index=std::distance(reordered.joint_names.begin(),found);named_target[named_index]=.5;
 run_for(mapped,named_target,500);check(std::abs(mapped.read().position_rad[named_index]-.5)<.15,"描述重排后仍按语义名映射状态与ctrl");mapped.close();
 SharpaDevice contact(assets,descriptor(assets,false),true);contact.connect();contact.enable();contact.home();
 JointArray<22> grip{};for(int i:{5,9,13,18})grip[i]=1.0;for(int i:{7,11,15,20})grip[i]=1.1;for(int i:{8,12,16,21})grip[i]=.8;
 run_for(contact,grip,1200);auto blocked=contact.snapshot();
 check(blocked.contact_count>0,"真实手-物体接触");double signed_load=0,max_load=0;
 for(int i:{5,7,8,9,11,12,13,15,16,18,20,21}){check(std::isfinite(blocked.contact_load_nm[i]),"接触力矩有限");signed_load+=blocked.contact_load_nm[i];max_load=std::max(max_load,std::abs(blocked.contact_load_nm[i]));}
 std::cout<<"抓握接触数="<<blocked.contact_count<<" 符号负载和="<<signed_load<<" 最大负载="<<max_load<<"\n";
 check(signed_load>.01&&max_load>.005,"阻碍闭合的环境反力对应正负载");
 bool blocked_motion=false;for(int i:{5,9,13,18})blocked_motion|=blocked.position_rad[i]<grip[i]-.1;check(blocked_motion,"物体阻碍实际关节运动");
 run_for(contact,JointArray<22>{},1200);auto released=contact.snapshot();double residual=0;for(double v:released.contact_load_nm)residual+=std::abs(v);
 check(residual<.02,"张开释放接触负载");
 contact.send(JointArray<22>{},now_ns()+1'000'000'000);std::this_thread::sleep_for(std::chrono::milliseconds(130));check(!contact.read().enabled[0],"原始100ms期限看门狗失能");contact.close();
 std::cout<<"Sharpa MuJoCo仿真回归通过\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
