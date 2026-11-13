#include "driver_impedance.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace motorhand::native;
void check(bool x, const char* s) { if (!x) throw std::runtime_error(s); }
bool reason(const DriverAxisObservation& axis,ImpedanceReason r){return axis.reasons&static_cast<std::uint64_t>(r);}
double torque(const DriverAxisCommand& c, double q, double dq) { return c.kp*(c.target_rad-q)-c.kd*dq+c.feedforward_nm; }
void axis_budget_and_withdrawal() {
    DriverImpedanceConfig cfg; bool damping = false;
    auto c = plan_driver_axis(1.0, 2.0, 0.0, 1, 60.0, 4.0, .03, cfg, damping);
    check(c.kp > 0 && c.kd > 0, "接触必须使用驱动端PD");
    check(torque(c,1,2) >= -3.4 && torque(c,1,2) <= 0, "弹簧和阻尼必须共同受名义预算约束");
    damping = true;
    c = plan_driver_axis(.3021, -.1, .3, 1, 40, 4, .03, cfg, damping);
    check(torque(c,.3021,-.1) <= 0, "已开阻尼的撤离不得产生向内拉力");
    c = plan_driver_axis(.3, -1, .3, 1, 60, 4, .03, cfg, damping);
    check(c.kp == 0 && c.kd == 0 && c.feedforward_nm == 0, "越过代理边界必须撤力");
    for (int sign : {-1,1}) for (double dq : {-20.,-2.,-.1,0.,.1,2.,20.}) {
        damping = true;
        c = plan_driver_axis(sign*.5,sign*dq,0,sign,60,2,.03,cfg,damping);
        const double t=sign*torque(c,sign*.5,sign*dq);
        check(t <= 1e-12 && t >= -1.7-1e-12, "正负轴的总PD必须单侧且有预算");
    }
}
void timestamp_velocity() {
    MotorVelocityEstimator estimator(12); GloveFrame g; g.received_ns.fill(1'000'000'000); g.oldest_ns=g.received_ns[0];
    auto v=estimator.update(g,1'000'000'000); check(v[0]==0,"首包不估速度");
    g.motor_rad[0]=.01; g.received_ns[0]+=10'000'000;
    v=estimator.update(g,1'010'000'000); check(v[0]>.5 && v[0]<.6,"速度应按新收包间隔低通");
    double previous=v[0]; v=estimator.update(g,1'020'000'000);
    check(v[0]==previous,"重复包不能衰减或更新速度");
    estimator.reset(); v=estimator.update(g,1'020'000'000); check(v[0]==0,"模式切换重置速度");
    v=estimator.update(g,1'200'000'000); check(v[0]==0,"过期速度必须重置");
    g.received_ns.fill(1'210'000'000);g.oldest_ns=g.received_ns[0];g.motor_rad[0]=.5;
    v=estimator.update(g,1'210'000'000); check(v[0]==0,"超期后首包不产生摩擦尖峰");
}
std::array<EncosMotorCalibration,12> calibration() {
    std::array<EncosMotorCalibration,12> c{};
    for(int i=0;i<12;++i)c[i]={i+1,0,1,1,-170,170,-2,2,i==1?2.:4.,true};return c;
}
void proxy_and_friction() {
    DriverImpedanceConfig cfg; cfg.ramp_s=0; DriverImpedancePlanner p(cfg,.3,calibration());
    GloveFrame g; g.oldest_ns=1'000'000'000;g.received_ns.fill(g.oldest_ns);g.motor_rad.fill(.5);
    GlovePose pose;for(int i=0;i<12;++i)pose.motor_jacobian[i*12+i]=1;
    std::array<double,240> jac{};for(int i=0;i<12;++i)jac[i*12+i]=1;
    ForceResult contact;contact.active=true;contact.contact[4]=true;contact.impedance_contact[4]=true;contact.contact_direction[4]=1;Vec20 target{};target[4]=.2;HandFrame h;
    p.reset(g.oldest_ns); auto r=p.step(target,h,g,pose,jac,contact,g.oldest_ns);
    check(r.command.kp[4]>0 && r.command.target_rad[4]<.5,"受阻行应建立局部代理");
    check(r.command.kp[3]==0 && r.command.kp[5]==0,"不能冻结全手");
    check(r.command.feedforward_nm[4]==0,"接触电机不得叠加摩擦");
    check(r.observation.proxy_valid[4]&&!r.observation.proxy_valid[3]&&r.observation.impedance_contact[4],"只有实际受阻轴与求解行可标记代理有效");
    check(r.observation.input_rad==g.motor_rad&&r.observation.source_ns==g.received_ns,"代理输入必须保留规划源快照");
    check(r.observation.axes[4].proxy_limited&&reason(r.observation.axes[4],ImpedanceReason::proxy_offset_limited)&&
          reason(r.observation.axes[4],ImpedanceReason::proxy_rate_limited),"偏移和参考速率受限必须记录实际分支");
    g.oldest_ns+=20'000'000;g.received_ns.fill(g.oldest_ns);g.motor_rad.fill(.52);
    r=p.step(target,h,g,pose,jac,contact,g.oldest_ns);
    check(r.command.feedforward_nm[0]>.20 && r.command.feedforward_nm[0]<=.25,"自由轴应补偿摩擦");
    check(r.command.feedforward_nm[1]>.08 && r.command.feedforward_nm[1]<=.10,"J2应使用较小摩擦");
    check(r.observation.filtered_velocity_rad_s==r.command.velocity_rad_s&&r.observation.filtered_velocity_rad_s[0]>0,"滤波速度必须来自本次规划");
    auto old_surface=r.proxy_surface_rad[4];target[4]=1;
    g.oldest_ns+=20'000'000;g.received_ns.fill(g.oldest_ns);
    r=p.step(target,h,g,pose,jac,contact,g.oldest_ns);
    check(std::abs(r.proxy_surface_rad[4]-old_surface)<=.0200001,"代理表面必须遵守更新速度");
    jac.fill(0);r=p.step(target,h,g,pose,jac,contact,g.oldest_ns);
    check(std::all_of(r.command.kp.begin(),r.command.kp.end(),[](double v){return v==0;}),"饱和后零Jac不得保留代理");
    check(std::all_of(r.command.feedforward_nm.begin(),r.command.feedforward_nm.end(),[](double v){return v==0;}),"无有效受阻方向不能误判为自由摩擦");
    check(!r.observation.proxy_valid[4]&&!r.observation.impedance_contact[4]&&
          reason(r.observation.axes[4],ImpedanceReason::unobservable_geometry),"失败规划必须撤销有效参考与参与mask并保留真实原因");
    contact.active=false;r=p.step(target,h,g,pose,jac,contact,g.oldest_ns);
    check(r.command.feedforward_nm[0]==0,"几何无效必须撤销摩擦");
}
void local_inverse_uses_motor_transmission() {
    DriverImpedanceConfig cfg;cfg.ramp_s=0;cfg.surface_rate=100;
    DriverImpedancePlanner planner(cfg,.3,calibration());GloveFrame glove;
    glove.oldest_ns=1'000'000'000;glove.received_ns.fill(glove.oldest_ns);glove.motor_rad.fill(.5);
    GlovePose pose;for(int i=0;i<12;++i)pose.motor_jacobian[i*12+i]=1;
    pose.motor_jacobian[0]=2;pose.motor_jacobian[1]=1;
    std::array<double,240> jac{};jac[0]=1;ForceResult result;result.active=true;result.contact[0]=true;result.impedance_contact[0]=true;result.contact_direction[0]=1;
    Vec20 target{};target[0]=.05;HandFrame hand;
    planner.reset(glove.oldest_ns);auto plan=planner.step(target,hand,glove,pose,jac,result,glove.oldest_ns);
    // A=[.5,-.5,...]，同一受阻误差应分配为两轴相反方向，而非按J或M^T走。
    check(plan.command.target_rad[0]<.5&&plan.command.target_rad[1]>.5,"局部逆必须使用J乘M逆并保持耦合方向");
    check(plan.command.kp[2]==0,"未参与受阻映射的轴不能冻结");
    jac[0]=0;plan=planner.step(target,hand,glove,pose,jac,result,glove.oldest_ns);
    check(!plan.valid&&plan.command.kp[0]==0,"最终目标饱和置零行后不能保留旧代理");
}
void contact_release_does_not_rebuild_reverse_proxy() {
    Vec12 signs{},scales{},lo{},hi{};signs.fill(1);scales.fill(1);lo.fill(-2);hi.fill(2);
    std::array<int,12> map{};for(int i=0;i<12;++i)map[i]=i+1;
    ForceController force("../config/feedback/wuji-contact-impedance-right.json","../config/feedback/wuji-max-torque-linear-right.json",signs,scales,map);
    force.set_motor_limits(lo,hi);std::uint64_t now=1'000'000'000;force.reset(now);
    DriverImpedanceConfig cfg;cfg.ramp_s=0;DriverImpedancePlanner planner(cfg,.3,calibration());planner.reset(now);
    GloveFrame glove;glove.motor_rad.fill(.5);GlovePose pose;pose.motor_rad=glove.motor_rad;
    std::array<double,240> jac{};for(int i=0;i<12;++i){pose.motor_jacobian[i*12+i]=1;jac[i*12+i]=1;}
    HandFrame hand;hand.has_velocity=hand.has_effort=true;hand.effort_a.fill(1.);
    Vec20 target{};target[4]=.2;ForceResult result;DriverImpedancePlan plan;
    auto step=[&](){now+=20'000'000;hand.received_ns=glove.oldest_ns=now;glove.received_ns.fill(now);
        pose.motor_rad=glove.motor_rad;result=force.step(target,hand,glove,pose,jac,pose.joint_rad,now,now);
        plan=planner.step(target,hand,glove,pose,jac,result,now);};
    for(int i=0;i<20;++i)step();check(result.contact[4]&&plan.command.kp[4]>0,"回归必须先进入真实接触PD");
    target[4]=.17;glove.motor_rad[4]=.49;step();
    check(result.contact[4]&&plan.command.kp[4]==0&&plan.command.feedforward_nm[4]==0,"主动回开时不能忽略控制器撤力判据");
    check(!plan.observation.impedance_contact[4]&&!plan.observation.proxy_valid[4]&&
          reason(plan.observation.axes[4],ImpedanceReason::coupled_release),"锁存contact不等于实际参与mask，回开无代理");
    target[4]=.2;for(int i=0;i<10;++i)step();
    target[4]=-.02;glove.motor_rad[4]=.48;step();
    check(result.contact[4],"释放迟滞期间旧contact应仍存在以禁止摩擦");
    check(plan.command.kp[4]==0&&plan.command.kd[4]==0&&plan.command.feedforward_nm[4]==0,"释放迟滞不能建立反向代理或启用摩擦");
}
void actual_used_mask_and_reference() {
    DriverImpedanceConfig cfg;cfg.ramp_s=0;cfg.surface_rate=100;cfg.max_proxy_offset=1.;
    DriverImpedancePlanner planner(cfg,1,calibration());GloveFrame g;
    g.oldest_ns=1'000'000'000;g.received_ns.fill(g.oldest_ns);g.motor_rad.fill(.5);
    GlovePose pose;std::array<double,240> jac{};for(int i=0;i<12;++i){pose.motor_jacobian[i*12+i]=1;jac[i*12+i]=1;}
    ForceResult f;f.active=true;f.contact[0]=f.impedance_contact[0]=true;f.contact_direction[0]=1;
    f.contact[1]=f.impedance_contact[1]=true;f.contact_direction[1]=-1;
    Vec20 target{};target[0]=target[1]=.5;HandFrame hand;planner.reset(g.oldest_ns);
    auto plan=planner.step(target,hand,g,pose,jac,f,g.oldest_ns);
    check(plan.observation.impedance_contact[0]&&!plan.observation.impedance_contact[1],"方向不符的候选行不能冒充实际参与行");
    check(plan.observation.proxy_valid[0]&&!plan.observation.proxy_valid[1],"逆解方向缺失时代理必须无效");
    check(plan.proxy_surface_rad[0]<plan.command.target_rad[0]&&plan.command.target_rad[0]<g.motor_rad[0]&&
          plan.observation.axes[0].torque_limited,"预算受限时代理参考必须与最终目标分别保留");
    jac[12]=jac[13]=0;plan=planner.step(target,hand,g,pose,jac,f,g.oldest_ns);
    check(!plan.valid&&std::none_of(plan.observation.impedance_contact.begin(),plan.observation.impedance_contact.end(),[](bool x){return x;}),
          "后续候选行不可观测导致整次释放时，不能保留先前已加入行");
    DriverAxisObservation observed;bool damping=true;
    const auto axis=plan_driver_axis(.3021,-.1,.3,1,40,4,.03,cfg,damping,&observed);
    check(axis.kp==0&&reason(observed,ImpedanceReason::planner_release_guard),"规划端预测撤力必须记录实际保护分支");
}
int main(){try{axis_budget_and_withdrawal();timestamp_velocity();proxy_and_friction();local_inverse_uses_motor_transmission();contact_release_does_not_rebuild_reverse_proxy();actual_used_mask_and_reference();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
