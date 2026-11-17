#include "session.hpp"
#include "common/session.hpp"
#include "hands/wuji/device_adapter.hpp"
#include "hands/wuji/compat.hpp"
#include "hands/wuji/wuji_retarget.hpp"
#include "encos_driver.hpp"
#include "trajectory.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
namespace motorhand::native {
namespace {
nlohmann::json read_config(const std::filesystem::path& path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("无法读取Wuji配置: "+path.string());
    return nlohmann::json::parse(f);
}
SessionTick legacy_tick(const ControlTick<20>& t) {
    SessionTick r;
    r.hand=legacy_wuji_frame(t.hand);
    r.glove=t.glove;
    r.wuji_target=t.hand_target;
    r.glove_target=t.glove_target;
    r.takeover=t.takeover;
    r.inverse_fresh=t.inverse_fresh;
    r.force_geometry_valid=t.force_geometry_valid;
    r.target_deadline_ns=t.target_deadline_ns;
    r.glove_pose=t.glove_pose;
    r.virtual_hand_nm=t.virtual_hand_nm;
    r.contact=t.contact;
    r.contact_phase=t.contact_phase;
    r.force_commanded=t.force_commanded;
    r.jacobian_source_ns=t.jacobian_source_ns;
    r.glove_output_source_ns=t.glove_output_source_ns;
    r.cycle_started_ns=t.cycle_started_ns;
    r.wuji_submit_ns=t.hand_submit_ns;
    r.glove_submit_ns=t.glove_submit_ns;
    r.cycle_completed_ns=t.cycle_completed_ns;
    r.linear_force=t.linear_force;
    r.wuji_load_fraction=t.load_fraction;
    r.feedback_input_hand=legacy_wuji_frame(t.feedback_input_hand);
    r.feedback_input_target=t.feedback_input_target;
    r.feedback_request_id=t.feedback_request_id;
    r.glove_feedforward_nm=t.glove_feedforward_nm;
    r.glove_projected_nm=t.glove_projected_nm;
    r.glove_requested_nm=t.glove_requested_nm;
    r.filtered_current_a=t.filtered_load;
    r.contact_reflection_gain=t.contact_reflection_gain;
    r.hand_to_motor=t.hand_to_motor;
    r.wuji_projection_current_a=t.projection_load;
    r.wuji_projection_received_ns=t.projection_received_ns;
    r.driver_impedance=t.driver_impedance;
    r.glove_friction_nm=t.glove_friction_nm;
    r.glove_driver_pd_nm=t.glove_driver_pd_nm;
    r.glove_driver_target_rad=t.glove_driver_target_rad;
    r.glove_driver_kp=t.glove_driver_kp;
    r.glove_driver_kd=t.glove_driver_kd;
    r.feedback_hardness=t.feedback_hardness;
    r.glove_driver_estimate_received_ns=t.glove_driver_estimate_received_ns;
    r.glove_driver_velocity_received_ns=t.glove_driver_velocity_received_ns;
    r.glove_impedance_observation=t.glove_impedance_observation;
    r.contact_evaluated_ns=t.contact_evaluated_ns;
    r.impedance_planned_ns=t.impedance_planned_ns;
    return r;
}
}
struct Session::Impl {
    SessionOptions options;WujiIdentity identity;
    std::unique_ptr<ControlSession<20>> control;
    bool trajectory_rejected{},closed{},connected{},enabled{};
    explicit Impl(SessionOptions o):options(std::move(o)) {
        validate_control_options(options);
    }
};
Session::Session(SessionOptions options):impl_(std::make_unique<Impl>(std::move(options))){}
Session::~Session()=default;
const WujiIdentity& Session::connect() {
    auto& s=*impl_;if(s.closed||s.connected||s.control)throw std::runtime_error("session cannot connect in current state");
    try {
        s.identity=inspect_wuji(s.options.wuji_serial,s.options.left);
        if((!s.options.wuji_serial.empty()&&s.identity.serial!=s.options.wuji_serial)||s.identity.left!=s.options.left||
            s.identity.firmware!=s.options.approved_firmware)throw std::runtime_error("Wuji preflight identity or firmware mismatch");
        std::unique_ptr<HandFeedback<20>> force;
        if(s.options.force_feedback) {
            const auto cfg=read_config(s.options.linear_feedback);
            if(cfg.at("wuji_serial_number").get<std::string>()!=s.identity.serial||
                cfg.at("glove_hand").get<std::string>()!=(s.options.left?"left":"right"))throw std::runtime_error("force feedback serial or hand mismatch");
            EncosDriver calibration(s.options.glove_mapping.string(),s.options.linear_feedback.string());
            Vec12 signs{},scales{};std::array<int,12> order{};
            const auto motors=calibration.calibration();
            for(int i=0;i<12;++i){const int raw=motors[i].raw_motor_id-1;if(raw<0||raw>=12)throw std::runtime_error("invalid Encos calibration");
                signs[raw]=motors[i].sign;scales[raw]=motors[i].scale;order[raw]=i+1;}
            force=std::make_unique<ForceController>(s.options.contact_feedback.string(),s.options.linear_feedback.string(),signs,scales,order,s.options.linear_force_feedback);
        }
        s.control=std::make_unique<ControlSession<20>>(s.options,std::make_unique<WujiHandDevice>(s.identity,s.options.approved_firmware),
            std::make_unique<WujiRetarget>(s.options.retarget_yaml,s.options.left),std::move(force));
        s.control->connect();s.connected=true;return s.identity;
    } catch(...) {s.closed=true;throw;}
}
HandFrame Session::read_wuji_state(){if(!impl_->control)throw std::runtime_error("session is not connected");auto frame=impl_->control->read_hand_state();
    if(!frame.has_velocity||!frame.has_load){impl_->control->stop();throw std::runtime_error("Wuji velocity or current feedback unavailable");}
    return legacy_wuji_frame(frame);}
void Session::validate_trajectory(const Trajectory& t) const {
    auto& s=*impl_;if(!s.connected||s.enabled||s.closed)throw std::runtime_error("trajectory validation requires connected, disabled session");
    s.trajectory_rejected=true;
    if(t.hand()!=(s.identity.left?"left":"right")||t.serial()!=s.identity.serial||t.firmware()!=s.identity.firmware)
        throw std::runtime_error("trajectory identity or firmware mismatch");
    const auto& d=s.control->descriptor();t.validate_limits(d.lower_rad,d.upper_rad);s.trajectory_rejected=false;
}
void Session::enable(){auto& s=*impl_;if(!s.control||s.trajectory_rejected)throw std::runtime_error("session cannot enable");s.control->enable();s.enabled=true;}
void Session::request_takeover(bool takeover){if(!impl_->control)throw std::runtime_error("session is not connected");impl_->control->request_takeover(takeover);}
SessionTick Session::step_inference(const Vec20& q,std::uint64_t stamp){if(!impl_->control)throw std::runtime_error("session is not connected");return legacy_tick(impl_->control->step_inference(q,stamp));}
SessionTick Session::step_takeover(){if(!impl_->control)throw std::runtime_error("session is not connected");return legacy_tick(impl_->control->step_takeover());}
void Session::stop(){auto& s=*impl_;if(s.control)s.control->stop();s.closed=true;s.connected=false;s.enabled=false;}
}
