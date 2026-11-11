"""Wuji Hand 2 ROS joint naming used by the standalone adapter.

The Tianji workspace exposes the same semantic joints through ``wuji_hand2``.
This local definition keeps the ROS2 package usable without that workspace package;
input messages may still use either generic or side-prefixed names.
"""

from __future__ import annotations

COMMAND_JOINT_NAMES = (
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_mcp_abd", "index_pip", "index_dip",
    "middle_mcp_flex", "middle_mcp_abd", "middle_pip", "middle_dip",
    "ring_mcp_flex", "ring_mcp_abd", "ring_pip", "ring_dip",
    "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip",
)
NUM_JOINTS = len(COMMAND_JOINT_NAMES)
_INDEX = {name: index for index, name in enumerate(COMMAND_JOINT_NAMES)}


def joint_index(name: str, side: str) -> int | None:
    """Return the generic command index for a ROS joint name.

    Accepts the generic names emitted by this project and the common Wuji names
    prefixed with ``l_``/``r_``.  Wuji descriptions also spell four finger joints
    as ``<finger>_finger_<joint>``; those aliases are normalized here.
    """
    value = str(name)
    prefix = f"{side[0]}_"
    if value.startswith(prefix):
        value = value[len(prefix):]
    value = value.replace("_finger_", "_")
    return _INDEX.get(value)
