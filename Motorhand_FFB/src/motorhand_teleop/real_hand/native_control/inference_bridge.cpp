#include "inference_bridge.h"
#include "session.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

using namespace motorhand::native;
struct mh_inference { explicit mh_inference(SessionOptions o) : session(std::move(o)) {} Session session; };
namespace {
thread_local char error_text[1024]{};
void set_error(const char* text) noexcept { std::snprintf(error_text, sizeof(error_text), "%s", text); }
template<class F> int call(mh_inference* s, F f) noexcept {
    error_text[0] = '\0';
    try {
        if (!s) throw std::invalid_argument("null inference session");
        f(); return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) { set_error("unknown native inference error"); }
    if (s) {
        try { s->session.stop(); }
        catch (const std::exception& e) {
            char original[sizeof(error_text)];
            std::snprintf(original, sizeof(original), "%s", error_text);
            std::snprintf(error_text, sizeof(error_text), "%.480s; shutdown: %.480s", original, e.what());
        } catch (...) {
            char original[sizeof(error_text)];
            std::snprintf(original, sizeof(original), "%s", error_text);
            std::snprintf(error_text, sizeof(error_text), "%.800s; shutdown: unknown error", original);
        }
    }
    return -1;
}
const char* required(const char* p) { if (!p || !*p) throw std::invalid_argument("missing inference option"); return p; }
void state(const HandFrame& in, mh_inference_state& out) {
    if (!in.has_velocity || !in.has_effort) throw std::runtime_error("Wuji velocity or current feedback unavailable");
    std::copy(in.position_rad.begin(), in.position_rad.end(), out.positions_rad);
    std::copy(in.velocity_rad_s.begin(), in.velocity_rad_s.end(), out.velocities_rad_s);
    std::copy(in.effort_a.begin(), in.effort_a.end(), out.currents_a);
    out.received_ns = in.received_ns;
}
void tick(const SessionTick& in, mh_inference_tick& out) {
    out = {};
    state(in.hand, out.hand);
    std::copy(in.wuji_target.begin(), in.wuji_target.end(), out.wuji_target_rad);
    if (in.glove_submit_ns) std::copy(in.glove_target.begin(), in.glove_target.end(), out.glove_output);
    if (in.force_commanded) std::copy(in.virtual_hand_nm.begin(), in.virtual_hand_nm.end(), out.virtual_hand_nm);
    out.target_deadline_ns = in.target_deadline_ns; out.glove_output_source_ns = in.glove_output_source_ns;
    out.cycle_started_ns = in.cycle_started_ns; out.wuji_submit_ns = in.wuji_submit_ns;
    out.glove_submit_ns = in.glove_submit_ns; out.cycle_completed_ns = in.cycle_completed_ns;
    out.jacobian_source_ns = in.jacobian_source_ns;
    out.takeover = in.takeover; out.inverse_fresh = in.inverse_fresh;
    out.force_geometry_valid = in.force_geometry_valid; out.force_commanded = in.force_commanded;
    out.contact_phase = static_cast<int>(in.contact_phase);
    for (size_t i = 0; i < 20; ++i) out.contact[i] = in.contact[i];
}
void copy_string(char (&out)[256], const std::string& value) {
    if (value.size() >= sizeof(out)) throw std::runtime_error("identity field too long");
    std::copy(value.begin(), value.end(), out); out[value.size()] = '\0';
}
}
extern "C" {
const char* mh_inference_error(void) { return error_text; }
int mh_inference_create(const mh_inference_options* o, mh_inference** out) {
    error_text[0] = '\0';
    if (out) *out = nullptr;
    try {
        if (!o || !out) throw std::invalid_argument("null inference options or result");
        if (o->left != 0 && o->left != 1) throw std::invalid_argument("invalid hand side");
        SessionOptions options;
        options.asset_root = required(o->asset_root); options.retarget_yaml = required(o->retarget_yaml);
        options.glove_mapping = required(o->glove_mapping); options.linear_feedback = required(o->linear_feedback);
        options.contact_feedback = required(o->contact_feedback); options.wuji_serial = o->wuji_serial ? o->wuji_serial : "";
        options.approved_firmware = required(o->approved_firmware); options.left = o->left;
        options.transition_ns = o->transition_ns;
        *out = new mh_inference(std::move(options)); return 0;
    } catch (const std::exception& e) { set_error(e.what()); }
      catch (...) { set_error("unknown native inference error"); }
    return -1;
}
int mh_inference_connect(mh_inference* s, mh_inference_identity* out) { return call(s, [&] {
    if (!out) throw std::invalid_argument("null identity result");
    const auto& id = s->session.connect(); *out = {};
    copy_string(out->serial, id.serial); copy_string(out->address, id.address); copy_string(out->firmware, id.firmware); out->left = id.left;
}); }
int mh_inference_read(mh_inference* s, mh_inference_state* out) { return call(s, [&] {
    if (!out) throw std::invalid_argument("null state result"); state(s->session.read_wuji_state(), *out);
}); }
int mh_inference_start(mh_inference* s) { return call(s, [&] { s->session.enable(); }); }
int mh_inference_request_takeover(mh_inference* s, int value) { return call(s, [&] {
    if (value != 0 && value != 1) throw std::invalid_argument("invalid takeover mode"); s->session.request_takeover(value != 0);
}); }
int mh_inference_step(mh_inference* s, const double* target, size_t count, uint64_t source_ns, mh_inference_tick* out) { return call(s, [&] {
    if (!target || count != 20 || !out) throw std::invalid_argument("inference requires exactly 20 targets and a result");
    Vec20 input; std::copy(target, target + 20, input.begin()); tick(s->session.step_inference(input, source_ns), *out);
}); }
int mh_inference_step_takeover(mh_inference* s, mh_inference_tick* out) { return call(s, [&] {
    if (!out) throw std::invalid_argument("null tick result"); tick(s->session.step_takeover(), *out);
}); }
int mh_inference_stop(mh_inference* s) { return call(s, [&] { s->session.stop(); }); }
void mh_inference_destroy(mh_inference* s) { try { delete s; } catch (...) {} }
}
