#ifndef MOTORHAND_INFERENCE_BRIDGE_H
#define MOTORHAND_INFERENCE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque session, single-threaded. All functions return 0 on success, -1 on
 * error; error text is available until the next call on the same thread.
 * A failed operation stops the session. No function enables it except start. */
typedef struct mh_inference mh_inference;
typedef struct {
    const char *asset_root, *retarget_yaml, *glove_mapping;
    const char *linear_feedback, *contact_feedback, *wuji_serial, *approved_firmware;
    int left;
    uint64_t transition_ns;
} mh_inference_options;
typedef struct { char serial[256], address[256], firmware[256]; int left; } mh_inference_identity;
typedef struct {
    double positions_rad[20], velocities_rad_s[20], currents_a[20];
    uint64_t received_ns;
} mh_inference_state;
typedef struct {
    mh_inference_state hand;
    double wuji_target_rad[20];
    /* Valid only when glove_submit_ns != 0: model motor radians in inference,
       nominal motor Nm in takeover, not joint angles or measured torque. */
    double glove_output[12];
    double virtual_hand_nm[20];
    uint64_t target_deadline_ns, glove_output_source_ns;
    uint64_t cycle_started_ns, wuji_submit_ns, glove_submit_ns, cycle_completed_ns;
    uint64_t jacobian_source_ns;
    int takeover, inverse_fresh, force_geometry_valid, force_commanded, contact_phase;
    unsigned char contact[20];
} mh_inference_tick;

const char *mh_inference_error(void);
int mh_inference_create(const mh_inference_options *, mh_inference **);
int mh_inference_connect(mh_inference *, mh_inference_identity *);
int mh_inference_read(mh_inference *, mh_inference_state *);
int mh_inference_start(mh_inference *);
int mh_inference_request_takeover(mh_inference *, int takeover);
/* source_ns is the original CLOCK_MONOTONIC model-production timestamp. */
int mh_inference_step(mh_inference *, const double *target, size_t count, uint64_t source_ns, mh_inference_tick *);
int mh_inference_step_takeover(mh_inference *, mh_inference_tick *);
int mh_inference_stop(mh_inference *);
void mh_inference_destroy(mh_inference *);
#ifdef __cplusplus
}
#endif
#endif
