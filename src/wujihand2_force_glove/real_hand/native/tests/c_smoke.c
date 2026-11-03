#include "real_hand.h"
#include <stdio.h>

/* 只验证构造/析构及C ABI；无scan/connect，不访问任何设备。 */
int main(void) {
    char error[512] = {0};
    RealHandOptions options = {
        .model = 2, .hand = 1, .serial_number = "离线构造，禁止连接",
        .address = "", .approved_firmware_version = "",
        .preflight_firmware_version = "",
        .max_velocity = 0, .max_acceleration = 0, .realtime_max_velocity = 0,
        .command_timeout = .5, .feedback_timeout = .5,
        .home_rate_hz = 100, .home_tolerance = .03,
        .home_timeout = 15, .enable_timeout = 5, .transition_duration = 1.5,
        .effort_limit = -1, .mit_kp = -1, .mit_kd = -1, .timeout_ms = 1000, .auto_home = 0
    };
    if (real_hand_abi_version() != 6) return 1;
    RealHand *hand = real_hand_create(&options, error, sizeof(error));
    if (!hand) { fprintf(stderr, "%s\n", error); return 1; }
    int status = real_hand_close(hand, error, sizeof(error));
    real_hand_destroy(hand);
    if (status) { fprintf(stderr, "%s\n", error); return 1; }
    puts("C ABI 离线构造与幂等清理通过，未连接设备");
    return 0;
}
