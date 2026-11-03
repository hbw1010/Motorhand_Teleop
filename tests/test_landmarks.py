from pathlib import Path

import numpy as np

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.landmarks import Ve33LandmarkSource


ROOT = Path(__file__).parents[1]


def test_ve33_source_emits_mediapipe_points_and_thumb_midpoint():
    source = Ve33LandmarkSource("right", asset_root=ROOT / "assets" / "hand_motor_ve33")
    frame = source.step({name: 0.0 for name in GLOVE_JOINTS})

    assert frame.points_m.shape == (21, 3)
    assert frame.jacobian_points_glove.shape == (21, 3, 12)
    assert np.all(np.isfinite(frame.points_m))
    assert np.allclose(frame.points_m[1], (frame.points_m[0] + frame.points_m[2]) * 0.5)
    assert len(frame.motor_positions) == 12
