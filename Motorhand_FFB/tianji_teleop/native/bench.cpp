// How long one frame's computation takes: the glove's motors -> the vE3.3 model -> WujiRetarget's 20 targets.
//   motorhand_glove_bench <Motorhand root> [frames]
#include "ve33_model.hpp"
#include "wuji_retarget.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

using namespace motorhand::native;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const std::filesystem::path root = argc > 1 ? argv[1] : ".";
    const int frames = argc > 2 ? std::stoi(argv[2]) : 600;
    Ve33Model model(root / "assets" / "hand_motor_ve33", false);
    WujiRetarget retarget(root / "config" / "retargeting" / "official" / "right.yaml", false);
    std::vector<double> model_ms, solve_ms;
    std::optional<GlovePose> last;
    for (int k = 0; k < frames; ++k) {
        Vec12 joints{};  // the fingers close and open at ~1 Hz (100 frames a second)
        for (int j = 0; j < 12; ++j)
            joints[j] = (j == 1 ? 0.05 * std::sin(k / 20.0) : 0.5 * (0.5 - 0.5 * std::cos(2 * M_PI * k / 100.0 + 0.3 * j)));
        const auto motors = model.evaluate_joints(joints).motor_rad;
        auto a = Clock::now();
        auto pose = model.evaluate_motors(motors, last ? &*last : nullptr);
        auto b = Clock::now();
        auto target = retarget.solve(pose.points_m);
        auto c = Clock::now();
        last = pose;
        if (!std::isfinite(target[0])) return 1;
        model_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        solve_ms.push_back(std::chrono::duration<double, std::milli>(c - b).count());
    }
    auto report = [](const char* name, std::vector<double> v) {
        v.erase(v.begin(), v.begin() + 20);
        std::sort(v.begin(), v.end());
        std::printf("%-34s median %6.3f ms  p95 %6.3f ms  max %6.3f ms\n", name, v[v.size() / 2], v[v.size() * 95 / 100], v.back());
    };
    report("vE3.3 model (motors -> 21 points)", model_ms);
    report("WujiRetarget (21 points -> 20)", solve_ms);
    return 0;
}
