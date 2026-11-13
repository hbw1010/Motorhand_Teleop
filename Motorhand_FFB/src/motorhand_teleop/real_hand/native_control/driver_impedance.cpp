#include "driver_impedance.hpp"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
namespace motorhand::native {
namespace {
constexpr double pi=3.14159265358979323846;
bool fresh(std::uint64_t stamp,std::uint64_t now) {return stamp && stamp<=now && now-stamp<100'000'000;}
void nonnegative(double x) {if(!std::isfinite(x)||x<0)throw std::invalid_argument("驱动阻抗参数必须为有限非负数");}
}
DriverImpedanceConfig load_driver_impedance_config(const std::string& path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("无法读取驱动阻抗配置: "+path);
    const auto j=nlohmann::json::parse(f);DriverImpedanceConfig c;
    const auto& d=j.at("driver_impedance"); const auto& r=j.at("friction_comp");
    c.stiffness_min=d.at("stiffness_nm_per_rad").at(0);c.stiffness_max=d.at("stiffness_nm_per_rad").at(1);
    c.gain=d.at("gain");c.damping_s=d.at("damping_s");c.damping_max=d.at("damping_max");
    c.joint_gain=d.at("joint_gain").get<Vec12>();c.kp_max=d.at("k_max_nm_per_rad").get<Vec12>();
    c.lookahead_s=d.at("release_lookahead_s");c.damping_off_velocity=d.at("damping_off_velocity_rad_s");
    c.damping_on_velocity=d.at("damping_on_velocity_rad_s");c.torque_margin=d.at("torque_margin");
    c.surface_rate=j.at("surface_rate_rad_s");c.inverse_regularization=j.at("inverse_regularization");
    c.max_proxy_offset=j.at("max_proxy_offset_rad");c.ramp_s=j.at("feedback_ramp_s");c.velocity_filter_hz=j.at("velocity_filter_hz");
    c.friction_enabled=r.at("enabled");c.coulomb_nm=r.at("coulomb_nm").get<Vec12>();c.viscous=r.at("viscous_nm_s_per_rad");
    c.velocity_scale=r.at("velocity_scale_rad_s");c.velocity_deadband=r.at("velocity_deadband_rad_s");
    c.friction_gain=r.at("gain");c.friction_max=r.at("max_nm");return c;
}
DriverAxisCommand plan_driver_axis(double q,double dq,double surface,int direction,double k,double limit,
    double reserve,const DriverImpedanceConfig& c,bool& damping,DriverAxisObservation* observation) {
    DriverAxisObservation unused; auto& observed=observation?*observation:unused;
    DriverAxisCommand out;out.target_rad=q;
    const double depth=direction*(q-surface), press=direction*dq;
    if(!direction||k<=0||depth<=0){
        observed.add(!direction?ImpedanceReason::free_axis:k<=0?ImpedanceReason::zero_stiffness:ImpedanceReason::outside_proxy);
        damping=false;return out;
    }
    if(damping && press < -c.damping_off_velocity)damping=false;
    else if(!damping && press > -c.damping_on_velocity)damping=true;
    const double budget=std::max(0.,c.torque_margin*limit-reserve);
    double b=damping?std::min(c.damping_max,c.damping_s*k):0;
    const double requested_damping=b;
    // 阻尼本身也可能耗尽预算；先减kd，再分配剩余弹簧预算。
    if(press>0)b=std::min(b,budget/press);
    double reach=std::min(depth,std::max(0.,budget-b*std::max(0.,press))/k);
    if(b<requested_damping || reach<depth) {
        observed.torque_limited=true;observed.add(ImpedanceReason::torque_budget_limited);
    }
    if(press<0)reach+=press*c.lookahead_s;
    // 总输出在当前及预测撤离姿态均不能朝向代理内部；否则完整撤力。
    const double total=-k*reach-b*press;
    const double predicted=total-k*std::min(0.,press)*c.lookahead_s;
    if(reach<=0||total>=0||predicted>=0){
        observed.add(ImpedanceReason::planner_release_guard);damping=false;return out;
    }
    observed.add(ImpedanceReason::contact_pd);
    out.target_rad=q-direction*reach;out.kp=k;out.kd=b;return out;
}
MotorVelocityEstimator::MotorVelocityEstimator(double hz):cutoff_(hz){nonnegative(hz);}
void MotorVelocityEstimator::reset(){q_={};velocity_={};stamps_={};}
Vec12 MotorVelocityEstimator::update(const GloveFrame& g,std::uint64_t now){
    for(int i=0;i<12;++i){const auto t=g.received_ns[i];
        if(!fresh(t,now)||!std::isfinite(g.motor_rad[i])){stamps_[i]=0;velocity_[i]=0;continue;}
        if(t==stamps_[i])continue;
        if(!stamps_[i]||t<stamps_[i]||t-stamps_[i]>=100'000'000)velocity_[i]=0;
        else {double dt=(t-stamps_[i])*1e-9;double raw=(g.motor_rad[i]-q_[i])/dt;
            velocity_[i]+=-std::expm1(-2*pi*cutoff_*dt)*(raw-velocity_[i]);}
        stamps_[i]=t;q_[i]=g.motor_rad[i];
    }return velocity_;
}
DriverImpedancePlanner::DriverImpedancePlanner(DriverImpedanceConfig c,double hardness,const std::array<EncosMotorCalibration,12>& cal)
:config_(c),hardness_(hardness),calibration_(cal),velocity_(c.velocity_filter_hz){
    for(double x:{c.stiffness_min,c.stiffness_max,c.gain,c.damping_s,c.damping_max,c.lookahead_s,c.damping_off_velocity,
        c.damping_on_velocity,c.torque_margin,c.surface_rate,c.inverse_regularization,c.max_proxy_offset,c.ramp_s,
        c.velocity_deadband,c.velocity_scale,c.friction_gain,c.friction_max,c.viscous})nonnegative(x);
    if(!std::isfinite(hardness)||hardness<0||hardness>1||c.torque_margin>1||c.torque_margin<=0||c.stiffness_max<c.stiffness_min||
        c.velocity_scale<=0||c.inverse_regularization<=0||c.surface_rate<=0||c.max_proxy_offset<=0||
        c.damping_off_velocity<c.damping_on_velocity)throw std::invalid_argument("驱动阻抗参数范围非法");
    std::array<bool,12> used{};
    for(int i=0;i<12;++i){const auto& m=cal[i];int raw=m.raw_motor_id-1;
        if(raw<0||raw>=12||used[raw]||!m.has_torque_limit||!std::isfinite(m.sign*m.scale)||m.sign*m.scale==0||
           !std::isfinite(m.raw_torque_limit_nm)||m.raw_torque_limit_nm<=0||!std::isfinite(m.model_min_rad)||
           !std::isfinite(m.model_max_rad)||m.model_min_rad>=m.model_max_rad)throw std::invalid_argument("驱动阻抗校准或限额非法");
        used[raw]=true;nonnegative(c.joint_gain[raw]);nonnegative(c.kp_max[raw]);nonnegative(c.coulomb_nm[raw]);
    }
}
void DriverImpedancePlanner::reset(std::uint64_t now){velocity_.reset();direction_={};damping_={};surface_={};reset_ns_=now;source_ns_=0;}
DriverImpedancePlan DriverImpedancePlanner::step(const Vec20& target,const HandFrame& hand,const GloveFrame& g,const GlovePose& pose,
    const std::array<double,240>& jac,const ForceResult& contact,std::uint64_t now){
    DriverImpedancePlan out;out.command.target_rad=g.motor_rad;
    auto& observation=out.observation;observation.input_rad=g.motor_rad;observation.source_ns=g.received_ns;
    auto release=[&](ImpedanceReason reason){
        observation.impedance_contact={};for(auto& axis:observation.axes)axis.add(reason);
        reset(now);return out;
    };
    if(!contact.active)return release(ImpedanceReason::inactive_geometry);
    for(int i=0;i<12;++i)if(!fresh(g.received_ns[i],now)||!std::isfinite(g.motor_rad[i]))return release(ImpedanceReason::invalid_input);
    using Mat12=Eigen::Matrix<double,12,12,Eigen::RowMajor>;using Mat20=Eigen::Matrix<double,20,12,Eigen::RowMajor>;
    const Eigen::Map<const Mat12> m(pose.motor_jacobian.data());const Eigen::Map<const Mat20> j(jac.data());
    if(!m.allFinite()||!j.allFinite()||j.norm()<1e-8)
        return release(!m.allFinite()||!j.allFinite()?ImpedanceReason::invalid_geometry:ImpedanceReason::unobservable_geometry);
    Eigen::FullPivLU<Mat12> lu(m);if(!lu.isInvertible()||lu.rcond()<1e-8)return release(ImpedanceReason::singular_transmission);
    const Mat20 a=j*lu.inverse();if(!a.allFinite())return release(ImpedanceReason::invalid_geometry);
    Eigen::Matrix<double,12,12> normal=Eigen::Matrix<double,12,12>::Identity()*config_.inverse_regularization;
    Eigen::Matrix<double,12,1> rhs=Eigen::Matrix<double,12,1>::Zero();
    bool any=false;
    for(int row=0;row<20;++row)if(contact.impedance_contact[row]){
        if(!std::isfinite(target[row])||!std::isfinite(hand.position_rad[row])||a.row(row).norm()<1e-8)return release(ImpedanceReason::invalid_contact_row);
        double e=target[row]-hand.position_rad[row];
        if(std::abs(contact.contact_direction[row])!=1 || contact.contact_direction[row]*e<=0)continue;
        normal+=a.row(row).transpose()*a.row(row);rhs+=a.row(row).transpose()*e;any=true;
        observation.impedance_contact[row]=true;
    }
    const Eigen::Matrix<double,12,1> delta=normal.ldlt().solve(rhs);if(!delta.allFinite())return release(ImpedanceReason::inverse_failed);
    out.command.velocity_rad_s=velocity_.update(g,now);
    observation.filtered_velocity_rad_s=out.command.velocity_rad_s;
    out.command.velocity_received_ns=g.received_ns;
    const auto source=g.oldest_ns;double dt=source_ns_&&source>=source_ns_?(source-source_ns_)*1e-9:.02;
    dt=std::clamp(dt,0.,.1);source_ns_=source;
    double ramp=config_.ramp_s>0?std::clamp(double(now>=reset_ns_?now-reset_ns_:0)*1e-9/config_.ramp_s,0.,1.):1.;
    for(int i=0;i<12;++i){const auto& cal=calibration_[i];const int raw=cal.raw_motor_id-1;
        auto& observed=observation.axes[i];
        double factor=std::abs(cal.sign*cal.scale),q=g.motor_rad[i],dq=out.command.velocity_rad_s[i];
        double cap=cal.raw_torque_limit_nm/factor;double k=std::min(config_.kp_max[raw],
            (config_.stiffness_min+(config_.stiffness_max-config_.stiffness_min)*hardness_)*config_.gain*config_.joint_gain[raw])*ramp;
        int dir=any && std::abs(delta[i])>1e-5?(delta[i]>0?1:-1):0;
        // 与受阻行有关但局部逆互相抵消的轴，保守关闭摩擦，不能把其当成自由空间。
        bool coupled=false;for(int row=0;row<20;++row)if(contact.contact[row]&&std::abs(a(row,i))>1e-7)coupled=true;
        DriverAxisCommand axis;axis.target_rad=q;
        if(dir){
            if(direction_[i]!=dir){surface_[i]=q;damping_[i]=false;}
            double desired=q-std::clamp(delta[i],-config_.max_proxy_offset,config_.max_proxy_offset);
            if(std::abs(delta[i])>config_.max_proxy_offset){observed.proxy_limited=true;observed.add(ImpedanceReason::proxy_offset_limited);}
            if(desired<cal.model_min_rad||desired>cal.model_max_rad){observed.motion_limited=true;observed.add(ImpedanceReason::soft_limit_target);}
            desired=std::clamp(desired,cal.model_min_rad,cal.model_max_rad);
            if(std::abs(desired-surface_[i])>config_.surface_rate*dt){observed.proxy_limited=true;observed.add(ImpedanceReason::proxy_rate_limited);}
            surface_[i]+=std::clamp(desired-surface_[i],-config_.surface_rate*dt,config_.surface_rate*dt);
            // 留出位置、增益、速度零码与tau_ff零码的最坏近似编码余量。
            const double reserve=.003/factor+k*factor*10./65535.+.062/factor/factor*std::abs(q-surface_[i])+
                                 .005/factor/factor*std::abs(dq)+.5*.003/factor;
            axis=plan_driver_axis(q,dq,surface_[i],dir,k,cap,reserve,config_,damping_[i],&observed);
        }else{
            surface_[i]=q;damping_[i]=false;
            observed.add(coupled?ImpedanceReason::coupled_release:ImpedanceReason::free_axis);
            if(!coupled&&!config_.friction_enabled)observed.add(ImpedanceReason::friction_disabled);
            if(!coupled&&config_.friction_enabled){const double moving=std::copysign(std::max(0.,std::abs(dq)-config_.velocity_deadband),dq);
                const double friction=config_.friction_gain*(config_.coulomb_nm[raw]/factor*std::tanh(moving/config_.velocity_scale)+config_.viscous*moving);
                if(std::abs(friction)>std::min(cap*.85,config_.friction_max/factor)){
                    observed.torque_limited=true;observed.add(ImpedanceReason::friction_limited);
                }
                axis.feedforward_nm=std::clamp(friction,-std::min(cap*.85,config_.friction_max/factor),std::min(cap*.85,config_.friction_max/factor))*ramp;
            }
        }
        direction_[i]=dir;out.proxy_surface_rad[i]=surface_[i];out.command.blocked_direction[i]=dir;
        observation.proxy_surface_rad[i]=surface_[i];observation.proxy_valid[i]=dir!=0;observation.blocking_direction[i]=dir;
        double nominal=axis.kp*(axis.target_rad-q)-axis.kd*dq+axis.feedforward_nm;
        if((q<=cal.model_min_rad&&nominal<0)||(q>=cal.model_max_rad&&nominal>0)){
            observed.motion_limited=true;observed.add(ImpedanceReason::soft_limit_release);axis={q,0,0,0};
        }
        out.command.target_rad[i]=axis.target_rad;out.command.kp[i]=axis.kp;out.command.kd[i]=axis.kd;
        out.command.feedforward_nm[i]=axis.feedforward_nm;out.nominal_nm[i]=axis.kp*(axis.target_rad-q)-axis.kd*dq+axis.feedforward_nm;
    }
    out.valid=true;return out;
}
} // namespace motorhand::native
