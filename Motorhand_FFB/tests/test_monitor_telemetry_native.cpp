#include "telemetry.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cmath>
using namespace motorhand::native;
using Json=nlohmann::json;
int main(){
 int fd=socket(AF_INET,SOCK_DGRAM,0);
 try{
  sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
  if(bind(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr)))throw std::runtime_error("bind失败");
  socklen_t n=sizeof(addr);getsockname(fd,reinterpret_cast<sockaddr*>(&addr),&n);
  timeval timeout{2,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
  auto receive=[&](){char b[32768];auto count=recv(fd,b,sizeof(b),0);if(count<=0)throw std::runtime_error("遥测未到达");return Json::parse(b,b+count);};
  TelemetryPublisher publisher(ntohs(addr.sin_port),"../hardware/encos_dexhand/configs/ffb/glove_mapping.json");
  SessionTick t{};t.hand.received_ns=t.glove.oldest_ns=123;t.glove.received_ns.fill(123);t.takeover=true;
  for(int i=0;i<68;++i)t.glove_pose.tree_rad[i]=i*.01;
  t.glove_pose.max_gap_m=1e-9;publisher.publish(t);
  auto p=receive();
  if(!p.contains("glove_proxy_valid") || p.at("glove_proxy_valid").size()!=12 ||
     p.at("glove_proxy_surface_rad")[0]!=nullptr || p.at("glove_control_reason")[0]!="unknown" ||
     p.at("contact_evaluated_ns")!=nullptr || p.at("impedance_planned_ns")!=nullptr)
      throw std::runtime_error("缺少验证遥测契约，未完成输出不能制造代理参考或原因");
  std::cout<<p.dump()<<'\n';
  if(p.at("glove_tree_rad").size()!=68||p.at("glove_tree_rad")[67]!=.67||p.at("hand")!="right")throw std::runtime_error("完整树遥测错误");
  const auto clock_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  t.glove.current_received_ns.fill(clock_ns);t.glove.measured_current_a.fill(.7);
  t.glove.current_received_ns[3]=clock_ns-200'000'000;
  publisher.publish(t);auto current=receive();
  if(current.at("glove_current_a")[3]!=nullptr || current.at("glove_current_received_ns")[3]!=clock_ns-200'000'000 ||
     current.at("glove_current_a")[0]!=.7 || current.at("glove_current_received_ns")[0]!=clock_ns)
      throw std::runtime_error("电流必须保留独立源时间，过期不能沿用位置时间显示");
  t.glove.current_received_ns.fill(clock_ns);
  t.force_commanded=true;t.force_geometry_valid=true;t.glove_submit_ns=125;
  t.contact_phase=ContactPhase::free;
  t.glove_projected_nm.fill(1.0);t.glove_requested_nm.fill(.5);t.glove_target.fill(.49);
  t.hand.received_ns=200;t.hand.has_effort=true;t.hand.effort_a.fill(.7);
  t.wuji_projection_received_ns=150;t.wuji_projection_current_a.fill(.3);
  publisher.publish(t);auto stages=receive();
  if(stages.at("glove_projected_torque_nm")[3]!=-1.0 ||
     stages.at("glove_requested_torque_nm")[3]!=-.5 || stages.at("glove_target")[3]!=-.49)
      throw std::runtime_error("不同阶段力矩必须在同一raw电机方向转换后分别发布");
  if(stages.at("wuji_projection_received_ns")!=150 || stages.at("wuji_projection_current_a")[0]!=.3 ||
     stages.at("wuji_current_a")[0]!=.7)
      throw std::runtime_error("投影必须携带配对电流与时间，不能替换成当前周期反馈");
  t.feedback_input_hand.received_ns=170;t.feedback_input_hand.has_effort=true;t.feedback_input_hand.has_velocity=true;
  t.feedback_input_hand.effort_a.fill(.4);t.feedback_input_hand.position_rad.fill(.2);t.feedback_input_hand.velocity_rad_s.fill(.1);
  t.feedback_input_target.fill(.5);t.feedback_request_id=7;t.glove_submit_ns=190;t.glove_feedforward_nm.fill(.125);
  publisher.publish(t);auto chain=receive();
  if(chain.at("wuji_feedback_received_ns")!=170 || chain.at("wuji_feedback_current_a")[0]!=.4 ||
     chain.at("feedback_request_id")!=7 || chain.at("glove_feedforward_torque_nm")[3]!=-.125 ||
     chain.at("wuji_current_a")[0]!=.7)
      throw std::runtime_error("控制链输入必须来自产生本次手套提交的同一请求，不能替换为最新状态");
  t.driver_impedance=true;t.glove_friction_nm.fill(.2);t.glove_driver_pd_nm.fill(-.7);
  t.glove_driver_target_rad.fill(.1);t.glove_driver_kp.fill(20);t.glove_driver_kd.fill(.16);
  publisher.publish(t);auto imp=receive();
  if(imp.at("feedback_control_mode")!="driver_impedance" || imp.at("glove_friction_torque_nm")[3]!=-.2 ||
     imp.at("glove_driver_pd_torque_nm")[3]!=.7 || imp.at("glove_driver_kp")[3]!=20 ||
     imp.at("glove_target_semantics")!="nominal_total_pd_at_submission")throw std::runtime_error("阻抗遥测模式或raw轴转换错误");
  t.glove_driver_estimate_received_ns.fill(151);
  t.glove_driver_velocity_received_ns.fill(149);
  t.contact_evaluated_ns=180;t.impedance_planned_ns=185;
  auto& observation=t.glove_impedance_observation;
  observation.proxy_surface_rad.fill(.3);observation.input_rad.fill(.5);observation.source_ns.fill(149);
  observation.filtered_velocity_rad_s.fill(.6);observation.blocking_direction.fill(1);observation.proxy_valid.fill(true);
  observation.proxy_valid[2]=false;observation.impedance_contact[4]=true;t.contact[5]=true;
  observation.axes[3].torque_limited=true;observation.axes[3].add(ImpedanceReason::torque_budget_limited);
  publisher.publish(t);auto validation=receive();
  Json mapping;{std::ifstream f("../hardware/encos_dexhand/configs/ffb/glove_mapping.json");f>>mapping;}
  const double zero=mapping["motors"]["4"]["zero_deg"].get<double>()*std::acos(-1.)/180.;
  auto near=[](double a,double b){return std::abs(a-b)<1e-12;};
  if(validation["glove_proxy_surface_rad"].size()!=12 || !near(validation["glove_proxy_surface_rad"][3],zero-.3) ||
     !near(validation["glove_proxy_input_rad"][3],zero-.5) || !near(validation["glove_driver_target_rad"][3],zero-.1) ||
     validation["glove_proxy_surface_rad"][2]!=nullptr || validation["glove_proxy_valid"][2]!=false ||
     validation["glove_filtered_velocity_rad_s"][3]!=-.6 || validation["glove_blocking_direction"][3]!=-1 ||
     validation["glove_proxy_source_ns"][3]!=149 || validation["glove_torque_limited"][3]!=true ||
     validation["glove_control_reason"][3]!="torque_budget_limited" || validation["impedance_contact"].size()!=20 ||
     validation["impedance_contact"][4]!=true || validation["impedance_contact"][5]!=false ||
     validation["contact_evaluated_ns"]!=180 || validation["impedance_planned_ns"]!=185)
      throw std::runtime_error("代理参考/最终目标/原始输入、反向轴、参与mask或源时间混淆");
  const double precise=-.12345678901234567;
  for(auto* a:{&t.glove_target,&t.glove_requested_nm,&t.glove_projected_nm,&t.glove_driver_pd_nm,
              &t.glove_driver_target_rad,&t.glove_driver_kp,&t.glove_driver_kd,&t.glove_friction_nm,
              &t.glove.raw_degrees,&t.glove.measured_current_a,&t.glove_pose.joint_rad})a->fill(precise);
  for(auto* a:{&t.hand.position_rad,&t.hand.velocity_rad_s,&t.hand.effort_a,&t.wuji_target,&t.virtual_hand_nm,
              &t.filtered_current_a,&t.wuji_projection_current_a,&t.wuji_load_fraction})a->fill(precise);
  t.hand.has_velocity=true;t.glove_pose.tree_rad.fill(precise);t.hand_to_motor.fill(precise);t.wuji_submit_ns=126;
  t.hand.received_ns=t.glove.oldest_ns=t.glove_output_source_ns=t.wuji_projection_received_ns=9'123'456'789'012'345'678ULL;
  t.glove.received_ns.fill(t.hand.received_ns);t.glove_driver_estimate_received_ns.fill(9'123'456'789'012'345'151ULL);
  t.glove_driver_velocity_received_ns.fill(9'123'456'789'012'345'149ULL);t.contact_reflection_gain=1.2345678901234567;
  observation.proxy_surface_rad.fill(precise);observation.input_rad.fill(precise);observation.filtered_velocity_rad_s.fill(precise);
  observation.source_ns.fill(9'123'456'789'012'345'148ULL);observation.proxy_valid.fill(true);
  for(auto& axis:observation.axes){axis.reasons=(1ULL<<25)-1;axis.torque_limited=axis.motion_limited=axis.proxy_limited=true;}
  publisher.publish(t);auto full=receive();
  if(full.at("glove_driver_estimate_received_ns")[3]!=9'123'456'789'012'345'151ULL || full.at("glove_driver_velocity_received_ns")[3]!=9'123'456'789'012'345'149ULL)
      throw std::runtime_error("PD状态与速度估计必须发布实际源时间");
  if(full.dump().size()>32768)throw std::runtime_error("完整阻抗遥测超过协议大小");
  std::cout<<"完整矩阵阻抗遥测大小="<<full.dump().size()<<"字节\n";
  t.hand_to_motor.fill(-1.2345678901234567e-123);
  publisher.publish(t);auto extended=receive();
  if(extended.dump().size()<=16384 || extended.dump().size()>32768 ||
     extended.at("wuji_to_glove_torque_matrix").size()!=240)
      throw std::runtime_error("大于16KiB的完整数值遥测必须无截断到达");
  std::cout<<"长科学计数矩阵遥测大小="<<extended.dump().size()<<"字节\n";
  // 非单位scale、反向和raw/model置换一起验证；只写临时映射，不触碰现场零位。
  const auto temporary=std::filesystem::temp_directory_path()/("telemetry-validation-"+std::to_string(getpid())+".json");
  std::swap(mapping["motors"]["1"]["joint"],mapping["motors"]["4"]["joint"]);
  mapping["motors"]["4"]["sign"]=-1.;mapping["motors"]["4"]["scale"]=2.;mapping["motors"]["4"]["zero_deg"]=30.;
  {std::ofstream f(temporary);f<<mapping;}
  {
    TelemetryPublisher transformed(ntohs(addr.sin_port),temporary);std::filesystem::remove(temporary);
    observation.input_rad[0]=.8;observation.proxy_surface_rad[0]=.6;observation.filtered_velocity_rad_s[0]=.4;
    observation.source_ns[0]=145;t.glove_driver_target_rad[0]=.2;t.glove_driver_kp[0]=3.;t.glove_driver_kd[0]=.2;
    transformed.publish(t);auto mapped=receive();const double z=std::acos(-1.)/6.;
    if(!near(mapped["glove_proxy_input_rad"][3],z-.4)||!near(mapped["glove_proxy_surface_rad"][3],z-.3)||
       !near(mapped["glove_driver_target_rad"][3],z-.1)||mapped["glove_filtered_velocity_rad_s"][3]!=-.2||
       mapped["glove_driver_kp"][3]!=12.||mapped["glove_driver_kd"][3]!=.8||mapped["glove_proxy_source_ns"][3]!=145)
        throw std::runtime_error("a=sign*scale、raw ID置换及含zero绝对rad转换错误");
  }
  t.force_commanded=false;publisher.publish(t);auto missing=receive();
  if(missing["glove_proxy_surface_rad"][0]!=nullptr || missing["glove_proxy_input_rad"][0]!=nullptr ||
     missing["glove_proxy_source_ns"][0]!=nullptr || missing["glove_filtered_velocity_rad_s"][0]!=nullptr ||
     missing["glove_control_reason"][0]!="unknown" || missing["impedance_contact"]!=nullptr)
      throw std::runtime_error("没有完成提交时不能发布残留代理输入和原因");
  publisher.status("fault","闭链求解失败");
  auto e=receive();
  if(e.at("type")!="teleop_status"||e.at("session_id")!=p.at("session_id")||e.at("message")!="闭链求解失败")throw std::runtime_error("事件错误");
  close(fd);std::cout<<"监视遥测离线验证通过\n";return 0;
 }catch(const std::exception& e){close(fd);std::cerr<<e.what()<<'\n';return 1;}
}
