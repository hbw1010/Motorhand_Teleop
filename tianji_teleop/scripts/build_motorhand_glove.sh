#!/usr/bin/env bash
# Build the C++ Motorhand glove worker for Tianji (tianji_teleop/native) in this checkout's Pixi env, from this
# checkout's own C++ (src/motorhand_teleop/real_hand/native_control): .cache/tianji-glove-native/motorhand_glove_native
# (and motorhand_glove_bench: how long a frame's computation takes).
#
#   bash tianji_teleop/scripts/build_motorhand_glove.sh      MOTORHAND=<checkout> (default: this one)
#
# Its env must be installed (`pixi install --frozen`). Tianji's motorhand_glove node runs it: its native_binary.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"  # tianji_teleop/
MOTORHAND="$(cd -- "${MOTORHAND:-$here/..}" && pwd -P)"
build="$MOTORHAND/.cache/tianji-glove-native"
[[ -f "$MOTORHAND/src/motorhand_teleop/real_hand/native_control/encos_driver.cpp" ]] || {
    echo "ERROR: no Motorhand C++ in $MOTORHAND" >&2; exit 1; }
mkdir -p "$build"
cd "$MOTORHAND"
exec "$HOME/.pixi/bin/pixi" run --frozen --no-install -e default bash -c '
cmake -S "$1/native" -B "$2" -DMOTORHAND_ROOT="$3" -DCMAKE_BUILD_TYPE=Release -Wno-dev \
    > "$2/configure.log" 2>&1 || { cat "$2/configure.log"; exit 1; }
cmake --build "$2" --parallel "$(nproc)"
' _ "$here" "$build" "$MOTORHAND"
