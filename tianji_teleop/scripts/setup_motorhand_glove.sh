#!/usr/bin/env bash
# On the PC a Motorhand exoskeleton glove plugs into: let this user open its CANable2 adapters (a udev rule, once,
# sudo), then read the glove for a few seconds, read only, and say whether it works.
#
#   bash tianji_teleop/scripts/setup_motorhand_glove.sh                 install the rule (asks for sudo), then check
#   bash tianji_teleop/scripts/setup_motorhand_glove.sh --check         only the check
#   bash tianji_teleop/scripts/setup_motorhand_glove.sh --check --side left --slot-ms 5 --rate 50 --seconds 15
#
# The check runs motorhand_glove's worker (the same as Tianji's glove:=motorhand, with
# tianji_teleop/config/motorhand_glove.yaml's slot_ms and rate_hz unless given) with this checkout
# (MOTORHAND=<dir> for another): no glove motor is enabled or driven, no Wuji hand is touched. Open
# and close the hand while it reads: it prints how far each finger's Wuji targets moved, how many motors answer and
# how old a frame's readings are when its target goes out. Stop Tianji's hands window (glove:=motorhand) first: the
# adapters take one program at a time.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"  # tianji_teleop/
RULE_SOURCE="$here/udev/90-motorhand-glove.rules"
RULE=/etc/udev/rules.d/90-motorhand-glove.rules
MOTORHAND="${MOTORHAND:-$(cd -- "$here/.." && pwd -P)}"
PYTHON="$MOTORHAND/.pixi/envs/default/bin/python"

setting() { sed -n "s/^ *$1: *\([0-9a-z.]*\).*/\1/p" "$here/config/motorhand_glove.yaml" | head -1; }
# The worker the config runs: the C++ one if built, else worker.py.
NATIVE="$MOTORHAND/.cache/tianji-glove-native/motorhand_glove_native"
if [[ "$(setting worker)" == native && -x "$NATIVE" ]]; then
    WORKER_CMD=("$NATIVE")
else
    WORKER_CMD=("$PYTHON" -I -u "$here/motorhand_glove/worker.py")
fi
CHECK_ONLY="" SIDE=right SLOT_MS="$(setting slot_ms)" RATE_HZ="$(setting rate_hz)" SECONDS_=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --side) SIDE="${2:?--side left|right}"; shift 2 ;;
        --slot-ms) SLOT_MS="${2:?--slot-ms MS}"; shift 2 ;;
        --rate) RATE_HZ="${2:?--rate HZ}"; shift 2 ;;
        --seconds) SECONDS_="${2:?--seconds S}"; shift 2 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (--help)" >&2; exit 2 ;;
    esac
done

adapters() { ls /dev/serial/by-id/ 2>/dev/null | grep -i canable2 | sed 's|^|/dev/serial/by-id/|' || true; }

mapfile -t ADAPTERS < <(adapters)
echo "CANable2 adapters: ${#ADAPTERS[@]} (a glove needs its 3)"
for a in "${ADAPTERS[@]}"; do echo "  $(basename "$a") -> $(readlink -f "$a")"; done
[[ ${#ADAPTERS[@]} -gt 0 ]] || { echo "ERROR: no CANable2 plugged in (lsusb | grep 16d0:117e)" >&2; exit 1; }

if [[ -z "$CHECK_ONLY" ]]; then
    if cmp -s "$RULE_SOURCE" "$RULE"; then
        echo "udev rule: already installed ($RULE)"
    else
        echo "udev rule: installing $RULE (sudo)"
        sudo install -m 0644 "$RULE_SOURCE" "$RULE"
    fi
    sudo udevadm control --reload
    for a in "${ADAPTERS[@]}"; do  # apply it to the adapters already plugged in: no replugging
        sudo udevadm trigger --action=add --name-match="$(readlink -f "$a")"
    done
    sudo udevadm settle
fi

ok=1
for a in "${ADAPTERS[@]}"; do
    d="$(readlink -f "$a")"
    if [[ -r "$d" && -w "$d" ]]; then
        echo "  $d: $(stat -c '%U:%G %a' "$d"), open for $(id -un)"
    else
        echo "  $d: $(stat -c '%U:%G %a' "$d"), NOT open for $(id -un) (without --check this installs the rule)"; ok=""
    fi
done
[[ -n "$ok" ]] || exit 1
devices=(); for a in "${ADAPTERS[@]}"; do devices+=("$(readlink -f "$a")"); done
busy="$(fuser "${devices[@]}" 2>/dev/null | xargs -r || true)"
if [[ -n "$busy" ]]; then
    echo "ERROR: the adapters are in use (pid $busy: $(ps -o comm= -p ${busy// /,} | xargs)); stop that first" >&2
    exit 1
fi
[[ -x "$PYTHON" ]] || { echo "ERROR: no Motorhand env at $PYTHON (pixi install --frozen in $MOTORHAND)" >&2; exit 1; }

out="$(mktemp)"; trap 'rm -f "$out"' EXIT
echo
echo "reading the $SIDE glove for $SECONDS_ s, read only (slot $SLOT_MS ms, $RATE_HZ Hz): open and close the hand a few times now"
cd /tmp
echo "worker: ${WORKER_CMD[0]##*/}"
timeout --foreground -k 5 -s TERM "$((SECONDS_ + 4))" "${WORKER_CMD[@]}" --side "$SIDE" --root "$MOTORHAND" \
    --slot-ms "$SLOT_MS" --rate "$RATE_HZ" --stats 2 >"$out" || true
/usr/bin/python3 - "$out" <<'PY'
import json, math, sys
lines = [json.loads(line) for line in open(sys.argv[1]) if line.strip().startswith("{")]
frames = [line["q"] for line in lines if "q" in line]
for line in lines:
    for kind in ("ready", "status"):
        if kind in line:
            print(f"{kind}: {line[kind]}")
stats = [line["stats"] for line in lines if "stats" in line]
if stats:
    s = stats[-1]
    motors = (f"motors read {s.get('motor_hz')} Hz (slowest {s.get('slowest_motor_hz')}); solve {s.get('solve_ms')} ms"
              if "motor_hz" in s else f"motors answer {s.get('reply_hz')} Hz of {s.get('expected_hz')} the slot allows")
    print(f"rate: {s.get('loop_hz')} targets/s; {motors}; a frame's readings {s.get('age_ms')} ms old when sent, "
          f"{s.get('span_ms')} ms apart")
if not frames:
    print("RESULT: no targets: see the status above"); sys.exit(1)
fingers = ("thumb", "index", "middle", "ring", "pinky")
print(f"{len(frames)} targets; how far each Wuji joint moved (deg, min..max):")
moved = 0
for f, finger in enumerate(fingers):
    cells = []
    for j in range(4):
        values = [math.degrees(q[4 * f + j]) for q in frames]
        lo, hi = min(values), max(values)
        moved += hi - lo > 10
        cells.append(f"j{j + 1} {lo:6.1f}..{hi:6.1f}")
    print(f"  {finger:6s} " + "   ".join(cells))
print("RESULT: the glove reads" + ("" if moved else "; but nothing moved more than 10 deg: did the hand move?"))
PY
