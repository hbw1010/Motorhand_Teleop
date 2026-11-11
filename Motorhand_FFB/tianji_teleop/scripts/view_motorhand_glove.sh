#!/usr/bin/env bash
# The Motorhand glove in MuJoCo, read only: the glove and the Wuji Hand 2 model following it, the hand drawn with
# Tianji's retarget settings (tianji_teleop/config/motorhand_glove.yaml's pinch), as the robot gets it. Needs a display
# (this PC's desktop; over ssh DISPLAY=:0). Close the window to end. Stop Tianji's hands window (glove:=motorhand) first:
# the glove's adapters take one program at a time.
#
#   bash tianji_teleop/scripts/view_motorhand_glove.sh             view
#   bash tianji_teleop/scripts/view_motorhand_glove.sh --zero      first zero all 12 motors at the pose the glove holds now (hold it
#                                                                  still), then view. After the glove lost power, zero
#                                                                  again: some joints come back 120 degrees off
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"  # tianji_teleop/
MOTORHAND="${MOTORHAND:-$(cd -- "$here/.." && pwd -P)}"
PYTHON="$MOTORHAND/.pixi/envs/default/bin/python"
setting() { sed -n "s/^ *$1: *\\([0-9.]*\\).*/\\1/p" "$here/config/motorhand_glove.yaml" | head -1; }
[[ -x "$PYTHON" ]] || { echo "ERROR: no Motorhand env at $PYTHON" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "ERROR: no display: run it on the PC's desktop, or DISPLAY=:0 over ssh" >&2; exit 1; }
ports=(); for a in /dev/serial/by-id/*CANable2*; do [[ -e "$a" ]] && ports+=("$(readlink -f "$a")"); done
busy="$(fuser "${ports[@]}" 2>/dev/null | xargs -r || true)"
[[ -z "$busy" ]] || { echo "ERROR: the glove's adapters are in use (pid $busy): stop that first" >&2; exit 1; }
cd "$MOTORHAND"
if [[ "${1:-}" == --zero ]]; then
    backup=".calibration_backup/$(date +%Y%m%d_%H%M%S)_before_zero"
    mkdir -p "$backup" && cp -p hardware/encos_dexhand/configs/glove_mapping.json hardware/encos_dexhand/configs/joint_calibration.json "$backup/"
    echo "zeroing all 12 motors at the pose the glove holds now (the old calibration: $backup)"
    DISPLAY= WAYLAND_DISPLAY= "$PYTHON" -I -m motorhand_teleop.encos_zero_step --all 2>&1 | grep -v -E "GLFW|warnings.warn|^$" || true
fi
exec "$PYTHON" -I -u "$here/motorhand_glove/viewer.py" --root "$MOTORHAND" \
    --pinch-d1 "$(setting pinch_d1_cm)" --pinch-d2 "$(setting pinch_d2_cm)" --pinch-weight "$(setting pinch_weight)"
