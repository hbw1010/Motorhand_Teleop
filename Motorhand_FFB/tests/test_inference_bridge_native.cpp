// Runs the actual inference C ABI and Session against simulated device boundaries.
#define main motorhand_session_test_main
#include "test_session_native.cpp"
#undef main
#include "motorhand_teleop/real_hand/native_control/inference_bridge.h"
#include <limits>

int main() {
    mh_inference* session = nullptr;
    try {
        Fixture fixture(std::filesystem::current_path());
        sim.reset(fixture.initial);
        sim.start();
        const auto assets = fixture.options.asset_root.string();
        const auto yaml = fixture.options.retarget_yaml.string();
        const auto mapping = fixture.options.glove_mapping.string();
        const auto linear = fixture.options.linear_feedback.string();
        const auto contact = fixture.options.contact_feedback.string();
        mh_inference_options options{assets.c_str(), yaml.c_str(), mapping.c_str(), linear.c_str(),
            contact.c_str(), serial, firmware, 0, 200'000'000};
        auto success = [](int result) {
            if (result != 0) throw std::runtime_error(mh_inference_error());
        };
        success(mh_inference_create(&options, &session));
        mh_inference_identity identity{};
        success(mh_inference_connect(session, &identity));
        check(std::string(identity.serial) == serial, "device identity mismatch");
        mh_inference_state state{};
        success(mh_inference_read(session, &state));
        check(state.received_ns && std::abs(state.positions_rad[0]-fixture.initial[0]) < 1e-9,
              "read did not return the measured hand state");
        success(mh_inference_start(session));
        mh_inference_tick tick{};
        success(mh_inference_step(session, fixture.initial.data(), 20, now_ns(), &tick));
        check(!tick.takeover && tick.wuji_submit_ns && tick.wuji_submit_ns < tick.target_deadline_ns,
              "inference command was not submitted with its deadline");
        success(mh_inference_request_takeover(session, 1));
        success(mh_inference_step_takeover(session, &tick));
        check(tick.takeover, "takeover did not activate");
        success(mh_inference_request_takeover(session, 0));
        success(mh_inference_step(session, fixture.initial.data(), 20, now_ns(), &tick));
        check(!tick.takeover, "inference did not resume");
        Vec20 bad = fixture.initial;
        bad[0] = std::numeric_limits<double>::quiet_NaN();
        check(mh_inference_step(session, bad.data(), 20, now_ns(), &tick) != 0,
              "invalid model output was accepted");
        no_expired_commands();
        { std::lock_guard<std::mutex> lock(sim.mutex);
          check(sim.hand_disable > 0 && sim.glove_disable > 0, "invalid model output did not release both devices"); }
        mh_inference_destroy(session); session = nullptr;
        sim.stop();
        sim.reset(fixture.initial); sim.start();
        success(mh_inference_create(&options, &session));
        success(mh_inference_connect(session, &identity));
        success(mh_inference_start(session));
        const auto expired = now_ns() - 100'000'000;
        check(mh_inference_step(session, fixture.initial.data(), 20, expired, &tick) != 0,
              "expired model output was accepted");
        no_expired_commands();
        { std::lock_guard<std::mutex> lock(sim.mutex);
          check(sim.hand_disable > 0 && sim.glove_disable > 0, "expired model output did not release both devices"); }
        mh_inference_destroy(session); session = nullptr;
        sim.stop();
        std::cout << "Native inference API integration passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (session) mh_inference_destroy(session);
        sim.stop();
        std::cerr << "Native inference API integration failed: " << error.what() << '\n';
        return 1;
    }
}
