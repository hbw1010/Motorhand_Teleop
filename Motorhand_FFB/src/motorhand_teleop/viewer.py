"""Read-only MuJoCo viewer for the glove and Wuji Hand 2 model.

Run with ``python -m motorhand_teleop.viewer --root <Motorhand_FFB>``.
The viewer reads glove feedback; it does not connect a Wuji device or drive glove motors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class SimulationGlovePose:
    """Apply the force-feedback simulation's mapping and closed-chain solver."""

    def __init__(self, mapping_path: Path, asset_root: Path) -> None:
        from glove_sim.kinematics import GloveKinematics
        from glove_sim.mapping import GloveMapping

        self.mapping = GloveMapping.load(mapping_path)
        self.kinematics = GloveKinematics(asset_root / "hand_motor_ve33_R", self.mapping)

    def solve(self, raw_angles: dict[int, float]) -> dict[str, float]:
        motors = np.asarray(self.mapping.raw_deg_to_model_rad(raw_angles), dtype=float)
        tree = self.kinematics.full_pose(motors, None, self.mapping)
        return dict(zip(self.kinematics.joint_names, map(float, tree)))


def apply_pinch(forward, *, d1_cm=None, d2_cm=None, weight=None) -> None:
    """Configure the Wuji optimizer's thumb-to-fingertip pinch distances and weight cap."""
    optimizer = forward.retargeter._core.optimizer
    if d1_cm:
        optimizer.d1 = np.full(4, float(d1_cm))
    if d2_cm:
        optimizer.d2 = np.full(4, float(d2_cm))
    if weight:
        cap, tips = float(weight), optimizer.MP_TIP_INDICES

        def pinch_alpha(keypoints):
            distances = np.linalg.norm(keypoints[tips[1:]] - keypoints[tips[0]], axis=1) * 100.0
            alphas = np.clip((optimizer.d2 - distances) / (optimizer.d2 - optimizer.d1 + 1e-8), 0.0, cap)
            return np.concatenate([[np.max(alphas)], alphas])

        optimizer._compute_pinch_alpha = pinch_alpha


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="the Motorhand_FFB checkout")
    parser.add_argument("--mapping", type=Path, default=None, help="the glove's glove_mapping.json")
    parser.add_argument("--pinch-d1", type=float, default=0.0)
    parser.add_argument("--pinch-d2", type=float, default=0.0)
    parser.add_argument("--pinch-weight", type=float, default=0.0)
    args = parser.parse_args(argv)
    from .encos_live_viewer import run_live_viewer

    pinch = {key: value for key, value in (("d1_cm", args.pinch_d1), ("d2_cm", args.pinch_d2),
                                           ("weight", args.pinch_weight)) if value}
    print(f"the Wuji hand drawn with pinch {pinch or 'Motorhand own'}; close the window to end", flush=True)
    try:
        run_live_viewer(args.root, mapping_path=args.mapping, pinch=pinch, match_simulation=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
