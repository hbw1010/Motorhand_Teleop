"""Standalone MuJoCo viewer for the glove and Wuji Hand 2 model.

It uses this checkout's retarget assets and is read-only: no Wuji device is
connected and no glove motor is driven. Run it with the project's Pixi Python:
``viewer.py --root <Motorhand_FFB>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this package, run by path in Motorhand's env
from motorhand_glove.worker import apply_pinch  # noqa: E402


def pinched(pipeline_class, **pinch):
    """pipeline_class whose from_yaml applies the pinch settings to what it builds."""

    class Pinched(pipeline_class):
        @classmethod
        def from_yaml(cls, *args, **kwargs):
            pipeline = pipeline_class.from_yaml(*args, **kwargs)
            apply_pinch(pipeline, **pinch)
            return pipeline

    return Pinched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="the Motorhand_FFB checkout")
    parser.add_argument("--mapping", type=Path, default=None, help="the glove's glove_mapping.json")
    parser.add_argument("--pinch-d1", type=float, default=0.0)
    parser.add_argument("--pinch-d2", type=float, default=0.0)
    parser.add_argument("--pinch-weight", type=float, default=0.0)
    args = parser.parse_args(argv)
    from motorhand_teleop import encos_live_viewer

    pinch = {key: value for key, value in (("d1_cm", args.pinch_d1), ("d2_cm", args.pinch_d2),
                                           ("weight", args.pinch_weight)) if value}
    encos_live_viewer.GloveWujiRetargetPipeline = pinched(encos_live_viewer.GloveWujiRetargetPipeline, **pinch)
    print(f"the Wuji hand drawn with pinch {pinch or 'Motorhand own'}; close the window to end", flush=True)
    try:
        encos_live_viewer.run_live_viewer(args.root, mapping_path=args.mapping)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
