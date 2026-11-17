#pragma once
#include "common/hand.hpp"
#include "hands/wuji/force_controller.hpp"
namespace motorhand::native {
inline HandState<20> common_wuji_frame(const HandFrame& h) {
    HandState<20> s;s.position_rad=h.position_rad;s.velocity_rad_s=h.velocity_rad_s;s.load=h.effort_a;
    s.has_velocity=h.has_velocity;s.has_load=h.has_effort;s.received_ns=h.received_ns;
    s.load_unit=HandLoadUnit::ampere;s.load_source="wuji-current";return s;
}
inline HandFrame legacy_wuji_frame(const HandState<20>& s) {
    return {s.position_rad,s.velocity_rad_s,s.load,s.has_velocity,s.has_load,s.received_ns};
}
inline FeedbackResult<20> common_wuji_feedback(const ForceResult& f) {
    FeedbackResult<20> r;
    r.virtual_hand_nm=f.virtual_hand_nm;r.load_fraction=f.wuji_load_fraction;r.model_motor_nm=f.model_motor_nm;
    r.contact=f.contact;r.impedance_contact=f.impedance_contact;r.contact_direction=f.contact_direction;
    r.active=f.active;r.linear=f.linear;r.phase=f.phase;r.projected_motor_nm=f.projected_motor_nm;
    r.filtered_load=f.filtered_current_a;r.reflection_gain=f.reflection_gain;r.hand_to_motor=f.hand_to_motor;
    r.projection_load=f.projection_current_a;r.projection_received_ns=f.projection_received_ns;return r;
}
}
