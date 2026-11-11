from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import SIM_ROOT, load_config
from .mapping import GloveMapping
from .objects import ObjectSpec
from .runner import Runner
from .sdk.fake import FakeGlove
from .sdk.real import RealGlove, RealGloveError, discover_ports
from .telemetry import Telemetry


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--feedback", action="store_true", help="启用力反馈（真机需显式确认）")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="秒；0=一直运行")
    parser.add_argument("--record", type=Path, default=None, help="自动记录目录")


def _mapping(config: dict, override: Path | None) -> GloveMapping:
    return GloveMapping.load(override or config["paths"]["glove_mapping"])


def _make_runner(args, *, real: bool) -> Runner:
    config = load_config(args.config)
    mapping = _mapping(config, args.mapping)
    specs = [ObjectSpec.from_dict(item) for item in config["objects"]["defaults"]]
    glove = (RealGlove(mapping, torque_sign=config["glove"].get("torque_sign"))
             if real else FakeGlove(mapping))
    telemetry = Telemetry(config, source="real" if real else "fake")
    runner = Runner(config, mapping, glove, specs, telemetry=telemetry, feedback=bool(args.feedback))
    if args.record:
        print(f"记录到: {runner.start_recording(args.record)}")
    return runner


def _headless(args, runner: Runner) -> int:
    try:
        runner.start()
    except Exception as error:
        print(f"启动失败: {error}")
        runner.close()
        return 2
    period = 1.0 / float(runner.config["simulation"].get("control_hz", 200.0))
    start = time.monotonic()
    last_print = start
    next_t = start
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            runner.tick(wall_clock=True, max_steps=30)
            # Fixed-rate schedule: sleep only the remainder of the period.
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
            if time.monotonic() - last_print >= 1.0 and runner.result is not None:
                result = runner.result
                print(
                    f"t={time.monotonic()-start:5.1f}s contacts={result.contact_count:3d} "
                    f"force={sum(result.finger_force_n.values()):7.2f}N "
                    f"tau_max={np.max(np.abs(result.torque_cmd_nm)):.3f}Nm "
                    f"loop={runner.stats.loop_hz:5.1f}Hz kin={result.kinematics_ms:.1f}ms "
                    f"tick={runner.stats.tick_ms:.1f}ms steps={runner.stats.steps_last_tick}"
                )
                last_print = time.monotonic()
    except KeyboardInterrupt:
        print("停止")
    finally:
        runner.close()
    return 0


def cmd_sim(args) -> int:
    runner = _make_runner(args, real=args.real)
    if args.headless or not args.ui:
        return _headless(args, runner)
    from .ui.app import run_ui
    return run_ui(runner)


def cmd_teleop(args) -> int:
    runner = _make_runner(args, real=True)
    if args.headless or not args.ui:
        return _headless(args, runner)
    from .ui.app import run_ui
    return run_ui(runner)


def cmd_check_glove(args) -> int:
    config = load_config(args.config)
    mapping = _mapping(config, args.mapping)
    print("CANable2 端口:")
    ports = discover_ports()
    print(ports if ports else "  (未发现 16d0:117e 设备)")
    if not ports:
        print("设备检查失败：插好三个 CANable2、安装 udev 规则并把用户加入 plugdev。")
        return 1
    glove = RealGlove(mapping)
    try:
        state = glove.open()
        print(f"online_ids={state.online_ids}")
        print(f"source={state.source} timestamp_ns={state.timestamp_ns}")
        print("motor  raw_deg  current_a  temp_c  error")
        for mid in range(1, 13):
            print(f"{mid:5d} {state.raw_deg[mid]:9.3f} {state.current_a[mid]:9.3f} "
                  f"{state.temperature_c[mid]:6.1f} {state.errors.get(mid, '')}")
        print("CHECK_OK（只读，无运动）")
        return 0
    except RealGloveError as error:
        print(f"CHECK_FAIL: {error}")
        return 2
    finally:
        glove.close()


def cmd_zero_glove(args) -> int:
    config = load_config(args.config)
    mapping_path = args.mapping or Path(config["paths"]["glove_mapping"])
    backup = args.backup_dir or Path(config["paths"]["zero_backup_dir"])
    from .sdk.zero import zero_online_glove
    try:
        return zero_online_glove(mapping_path, finger=args.finger, all_motors=args.all,
                                 backup_dir=backup, assume_yes=args.yes)
    except Exception as error:
        print(f"ZERO_FAIL: {error}")
        return 2


def cmd_home_glove(args) -> int:
    config = load_config(args.config)
    mapping_path = args.mapping or Path(config["paths"]["glove_mapping"])
    script = SIM_ROOT.parent / "src" / "motorhand_teleop" / "encos_home.py"
    if not script.is_file():
        print(f"HOME_FAIL: 缺少 {script}")
        return 2
    command = [sys.executable, str(script), "--mapping", str(mapping_path),
               "--tolerance-rad", str(args.tolerance_rad),
               "--speed-rpm", str(args.speed_rpm), "--timeout-s", str(args.timeout_s)]
    print("执行:", " ".join(command))
    rc = int(subprocess.call(command))
    if rc != 0:
        print(f"HOME_FAIL: 退出码 {rc}（检查 12 轴在线、适配器占用和报警）")
    return rc


def cmd_record(args) -> int:
    args.record = args.record or Path("data")
    return cmd_sim(args)


def cmd_replay(args) -> int:
    """Replay a JSONL/CSV recording into the offline simulation (headless)."""
    path = Path(args.file)
    if not path.exists():
        print(f"REPLAY_FAIL: 文件不存在 {path}")
        return 2
    rows = []
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        import csv
        with path.open(encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(line for line in stream if not line.startswith("#"))]
    if not rows:
        print("REPLAY_FAIL: 空文件")
        return 2
    config = load_config(args.config)
    mapping = _mapping(config, args.mapping)
    specs = [ObjectSpec.from_dict(item) for item in config["objects"]["defaults"]]
    runner = Runner(config, mapping, FakeGlove(mapping, demo=False), specs,
                    telemetry=Telemetry(config, source="replay", ), feedback=False)
    runner.start()
    sim = runner.sim
    for row in rows:
        q = [float(row.get(f"sim_joint_{i}_rad", row.get(f"real_joint_{i}_rad", 0.0)) or 0.0)
             for i in range(1, 13)]
        # FakeGlove is not used for replay pose; drive the kinematic glove directly.
        sim.update_glove(np.asarray(q))
        sim.step(feedback_enabled=False, dt_s=1.0 / float(config["simulation"]["control_hz"]))
    runner.close()
    print(f"REPLAY_OK rows={len(rows)} (no hardware output)")
    return 0


def _session_path(args) -> Path | None:
    from .recording import latest_session
    if args.session:
        return Path(args.session)
    config = load_config(getattr(args, "config", None))
    directory = Path(config.get("telemetry", {}).get("default_dir", "data"))
    if not directory.is_absolute():
        directory = Path(config["_sim_root"]) / directory
    return latest_session(directory)


def cmd_review(args) -> int:
    """Interactive review of a recorded session (curves + scene at a time cursor)."""
    path = _session_path(args)
    if path is None:
        print("REVIEW_FAIL: 没有找到记录（在仿真界面点“开始记录”，或用 --record）")
        return 2
    from .ui.review import run_review
    return run_review(path)


def cmd_export_session(args) -> int:
    from .recording import load_session
    path = _session_path(args)
    if path is None:
        print("EXPORT_FAIL: 没有找到记录")
        return 2
    session = load_session(path)
    out = session.to_csv(args.out)
    print(f"EXPORT_OK {len(session)} 帧 -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motorhand glove force-feedback simulation")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("sim", help="离线仿真（默认假手套）")
    _common(sim)
    sim.add_argument("--real", action="store_true", help="使用真实手套，而不是假手套")
    sim.add_argument("--ui", action="store_true", default=True, help=argparse.SUPPRESS)
    sim.set_defaults(func=cmd_sim)

    tele = sub.add_parser("teleop", help="真实手套遥操作 + 仿真")
    _common(tele)
    tele.add_argument("--ui", action="store_true", default=True, help=argparse.SUPPRESS)
    tele.set_defaults(func=cmd_teleop)

    check = sub.add_parser("check-glove", help="只读设备检查")
    check.add_argument("--config", type=Path, default=None)
    check.add_argument("--mapping", type=Path, default=None)
    check.set_defaults(func=cmd_check_glove)

    zero = sub.add_parser("zero-glove", help="软件零位标定（不发送位置/MIT）")
    zero.add_argument("--config", type=Path, default=None)
    zero.add_argument("--mapping", type=Path, default=None)
    zero.add_argument("--backup-dir", type=Path, default=None)
    zero.add_argument("--finger", choices=("thumb", "index", "middle", "ring", "pinky"), default=None)
    zero.add_argument("--all", action="store_true", help="一次采集 12 轴当前反馈")
    zero.add_argument("--yes", action="store_true", help="兼容非交互脚本")
    zero.set_defaults(func=cmd_zero_glove)

    home = sub.add_parser("home-glove", help="统一使能并回零（危险动作，需确认）")
    home.add_argument("--config", type=Path, default=None)
    home.add_argument("--mapping", type=Path, default=None)
    home.add_argument("--tolerance-rad", type=float, default=0.15)
    home.add_argument("--speed-rpm", type=float, default=5.0)
    home.add_argument("--timeout-s", type=float, default=30.0)
    home.set_defaults(func=cmd_home_glove)

    review = sub.add_parser("review", help="回放分析记录（曲线 + 当时的仿真/真手状态）")
    review.add_argument("session", nargs="?", default=None, help="记录目录；默认最新一次")
    review.add_argument("--config", type=Path, default=None)
    review.set_defaults(func=cmd_review)

    export = sub.add_parser("export-session", help="把记录导出为 CSV")
    export.add_argument("session", nargs="?", default=None, help="记录目录；默认最新一次")
    export.add_argument("--out", type=Path, default=None)
    export.add_argument("--config", type=Path, default=None)
    export.set_defaults(func=cmd_export_session)

    rec = sub.add_parser("record", help="仿真并记录")
    _common(rec)
    rec.add_argument("--real", action="store_true")
    rec.add_argument("--ui", action="store_true", default=True, help=argparse.SUPPRESS)
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("replay", help="回放 JSONL/CSV 到离线仿真")
    rep.add_argument("file", type=Path)
    rep.add_argument("--config", type=Path, default=None)
    rep.add_argument("--mapping", type=Path, default=None)
    rep.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
