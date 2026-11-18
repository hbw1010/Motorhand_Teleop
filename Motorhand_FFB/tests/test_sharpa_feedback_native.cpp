#include "hands/sharpa/feedback.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace motorhand::native;
using namespace motorhand::native::sharpa;
namespace {
constexpr std::uint64_t stamp = 2'000'000'000;
const std::filesystem::path wuji_config_dir = "../config/feedback";
std::array<EncosMotorCalibration, 12> calibration() {
  std::array<EncosMotorCalibration, 12> c{};
  for (int i = 0; i < 12; ++i) {
    c[i].raw_motor_id = 12 - i;
    c[i].sign = -1;
    c[i].scale = 2;
    c[i].raw_torque_limit_nm = 10;
    c[i].has_torque_limit = true;
  }
  return c;
}
struct Input {
  HandState<22> h;
  GloveFrame g;
  GlovePose p;
  RetargetJacobian<22> jac{};
  JointArray<22> target{};
  Vec12 reference{};
  std::uint64_t now = stamp;
  Input() {
    h.connected = true;
    h.enabled.fill(true);
    h.online.fill(true);
    h.has_load = h.has_velocity = true;
    h.load_unit = HandLoadUnit::newton_metre;
    h.load_source = torque_source;
    for (int i = 0; i < 12; ++i)
      p.motor_jacobian[i * 12 + i] = 2;
    jac[21 * 12] = 2;
    fresh();
  }
  void fresh() {
    h.received_ns = g.oldest_ns = now;
    g.received_ns.fill(now);
  }
  FeedbackResult<22> step(Feedback &f) {
    fresh();
    return f.step(target, h, g, p, jac, reference, now, now);
  }
};
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
void linear_last_axis_and_calibration(const std::string &path) {
  Feedback f(path, wuji_config_dir, true, calibration());
  Vec12 lo, hi;
  lo.fill(-2);
  hi.fill(2);
  f.set_motor_limits(lo, hi);
  f.reset(stamp - 1'000'000'000);
  Input in;
  in.h.load[21] = .189369;
  auto r = in.step(f);
  assert(r.active && r.linear);
  assert(std::abs(r.load_fraction[21] - 1) < 1e-10);
  assert(std::abs(r.hand_to_motor[21] - 1) < 1e-10);
  assert(r.projected_motor_nm[0] < -4.9 && r.projected_motor_nm[0] > -5);
  assert(r.model_motor_nm[0] == 0);
  assert(r.model_motor_nm[11] == 0);
  f.acknowledge_torque(r.model_motor_nm);
  in.now += 300'000'000;
  r = in.step(f);
  assert(r.model_motor_nm[0] < -4.9 && r.model_motor_nm[0] > -5);
  f.acknowledge_torque(r.model_motor_nm);
  in.now += 20'000'000;
  in.h.load[21] = -.189369;
  r = in.step(f);
  assert(r.projected_motor_nm[0] > 4.9);
}
void coupled_projection_cancels(const std::string &path) {
  Feedback f(path, wuji_config_dir, true, calibration());
  Vec12 lo, hi;
  lo.fill(-2);
  hi.fill(2);
  f.set_motor_limits(lo, hi);
  f.reset(stamp - 1'000'000'000);
  Input in;
  in.jac[0] = -2;
  in.h.load[0] = 3.3;
  in.h.load[21] = .189369;
  auto r = in.step(f);
  assert(std::abs(r.projected_motor_nm[0]) < 1e-10);
}
void missing_and_wrong_units_stop(const std::string &path) {
  for (int i = 0; i < 3; ++i) {
    Feedback f(path, wuji_config_dir, false, calibration());
    Vec12 lo, hi;
    lo.fill(-2);
    hi.fill(2);
    f.set_motor_limits(lo, hi);
    f.reset(stamp);
    Input in;
    if (i == 0)
      in.h.has_load = false;
    if (i == 1)
      in.h.load_unit = HandLoadUnit::ampere;
    if (i == 2)
      in.h.load_source = "net-external-torque";
    rejects([&] { in.step(f); });
  }
}
void contact_and_immediate_opening_release(const std::string &path) {
  Feedback f(path, wuji_config_dir, false, calibration());
  Vec12 lo, hi;
  lo.fill(-2);
  hi.fill(2);
  f.set_motor_limits(lo, hi);
  f.reset(stamp - 1'000'000'000);
  Input in;
  in.h.load[21] = .1;
  in.target[21] = .3;
  auto r = in.step(f);
  assert(!r.contact[21]);
  for (int i = 0; i < 12; ++i) {
    in.now += 20'000'000;
    r = in.step(f);
    f.acknowledge_torque(r.model_motor_nm);
  }
  assert(r.contact[21] && r.impedance_contact[21] &&
         r.contact_direction[21] == 1);
  assert(r.virtual_hand_nm[21] < 0 && r.model_motor_nm[0] < 0);
  in.target[21] = .2;
  in.now += 20'000'000;
  r = in.step(f);
  assert(!r.impedance_contact[21] && r.model_motor_nm[0] == 0);
}
void thumb_contact_survives_small_target_retreat(const std::string &path) {
  Feedback feedback(path, wuji_config_dir, false, calibration());
  Vec12 lower, upper;
  lower.fill(-2);
  upper.fill(2);
  feedback.set_motor_limits(lower, upper);
  feedback.reset(stamp - 1'000'000'000);
  Input input;
  input.jac[0] = 2;
  input.h.position_rad[0] = .03;
  input.h.load[0] = .33;
  input.h.velocity_rad_s[0] = .2;
  input.target[0] = .10;
  auto output = input.step(feedback);
  assert(!output.impedance_contact[0]);
  for (int frame = 0; frame < 2; ++frame) {
    input.now += 20'000'000;
    output = input.step(feedback);
  }
  assert(output.impedance_contact[0]);
  input.now += 20'000'000;
  input.target[0] -= .0005;
  assert(input.step(feedback).impedance_contact[0]);
  input.now += 20'000'000;
  input.target[0] -= .02;
  assert(!input.step(feedback).impedance_contact[0]);
}
void contact_uses_sharpa_motor_cap(const std::string &path) {
  const auto axes = calibration();
  Feedback feedback(path, wuji_config_dir, false, axes);
  Vec12 lower, upper;
  lower.fill(-2);
  upper.fill(2);
  feedback.set_motor_limits(lower, upper);
  feedback.reset(stamp - 1'000'000'000);
  Input input;
  input.jac[21 * 12] = 400;
  input.h.load[21] = .189369;
  input.target[21] = .3;
  FeedbackResult<22> output;
  for (int frame = 0; frame < 5; ++frame) {
    output = input.step(feedback);
    input.now += 20'000'000;
  }
  assert(axes[0].raw_motor_id == 12);
  assert(output.impedance_contact[21]);
  const double raw_torque = std::abs(output.model_motor_nm[0] * axes[0].scale);
  assert(raw_torque > 9.9 && raw_torque <= 10);
}
void simulator_source_is_explicit(const std::string &path) {
  Feedback real(path, wuji_config_dir, true, calibration());
  Feedback simulated(path, wuji_config_dir, true, calibration(),
                     "sharpa-mujoco-contact-joint-torque");
  Vec12 lo, hi;
  lo.fill(-2);
  hi.fill(2);
  real.set_motor_limits(lo, hi);
  simulated.set_motor_limits(lo, hi);
  real.reset(stamp);
  simulated.reset(stamp);
  Input in;
  in.h.load_source = "sharpa-mujoco-contact-joint-torque";
  rejects([&] { in.step(real); });
  assert(in.step(simulated).active);
  rejects([&] {
    Feedback invalid(path, wuji_config_dir, true, calibration(), "unverified-source");
  });
}

void stale_geometry_releases(const std::string &path) {
  Feedback f(path, wuji_config_dir, false, calibration());
  Vec12 lo, hi;
  lo.fill(-2);
  hi.fill(2);
  f.set_motor_limits(lo, hi);
  f.reset(stamp);
  Input in;
  in.p.motor_jacobian = {};
  rejects([&] { in.step(f); });
}
void absolute_start_waits_for_position_transition(const std::string &path) {
  for(bool linear:{false,true}) {
    Feedback f(path, wuji_config_dir, linear,calibration(),torque_source,500'000'000);
    Vec12 lo,hi;lo.fill(-2);hi.fill(2);f.set_motor_limits(lo,hi);f.reset(stamp);
    Input in;in.h.load[21]=.189369;
    for(int k=0;k<25;++k){auto r=in.step(f);assert(!r.active);for(double v:r.model_motor_nm)assert(v==0);f.acknowledge_torque(r.model_motor_nm);in.now+=20'000'000;}
    for(int k=0;k<5;++k){auto r=in.step(f);assert(r.active);f.acknowledge_torque(r.model_motor_nm);in.now+=20'000'000;}
  }
}
} // namespace
int main(int argc, char **argv) {
  const std::string path =
      argc > 1 ? argv[1] : "config/feedback/sharpa-right.json";
  simulator_source_is_explicit(path);
  linear_last_axis_and_calibration(path);
  coupled_projection_cancels(path);
  missing_and_wrong_units_stop(path);
  contact_and_immediate_opening_release(path);
  thumb_contact_survives_small_target_retreat(path);
  contact_uses_sharpa_motor_cap(path);
  stale_geometry_releases(path);
  absolute_start_waits_for_position_transition(path);
  std::cout << "Sharpa 22轴Nm反馈数学通过\n";
}
