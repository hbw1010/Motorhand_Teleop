"""One Motorhand exoskeleton glove -> Wuji Hand 2 targets, in the Motorhand_Teleop checkout's own Pixi env.

motorhand_glove.node starts one per glove. The forward chain is Motorhand's, as its own glove-to-hand2 teleop
(real_hand.runtime.PairRuntime) runs it: the glove's 12 encoders -> the vE3.3 closed chain -> 21 task points ->
the official WujiRetarget -> 20 targets, clipped to the retargeter's joint limits, the four fingers' spread at
zero (the glove has no spread input). The glove is only read: no motor is enabled, homed or driven.

The retarget (5-7 ms a frame, mostly Python) runs in a process of its own, Motorhand's ForwardRetargetWorker,
where the checkout has it (real_hand/forward_worker.py): in this process it holds the GIL the glove's CAN threads
need every slot, which halved the motors' replies on the real glove and let a motor go ~90 ms unread. Without it,
in this process (the ready line says which). A frame is not given a deadline: a slow one is late, not fatal.

One JSON object per line on stdout (anything Motorhand prints goes to stderr):
    {"ready": {...}}            connected, all 12 motors online
    {"ns": <ns>, "q": [20]}     a target, from the glove reading at monotonic time <ns>
    {"status": "<why>"}         not streaming (no adapter, a motor offline, a read failed); it reconnects every
                                --retry seconds and says so again only when the reason changes
    {"stats": {...}}            every --stats seconds while streaming: targets out a second (loop_hz), each
                                motor's CAN replies a second (reply_hz, its median; slowest_reply_hz) against what
                                the slot allows (expected_hz: 1 / (slot x the motors on its channel); in a simulation
                                with no lost replies a 1 ms slot gave ~80% of it, the threads' own overhead), how old
                                the oldest of a frame's 12 readings is when its target goes out (age_ms, the retarget
                                included) and how far apart they are (span_ms)

The glove's CAN threads ask one motor per --slot-ms on each adapter (Motorhand's own: 5 ms), so a motor is read
every slot x the motors on its adapter: 20 ms with three adapters of four motors.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

SPREAD = (5, 9, 13, 17)  # index..pinky mcp_abd in the 20 targets
MOTORS = tuple(range(1, 13))
# What the glove SDK prints while connecting that says why a connection failed (an adapter it could not open);
# the rest (every platform serial port it skipped, on every retry) would only flood the log.
SDK_REASONS = ("❌", "⚠️ [")


def pipeline(root: Path, side: str):
    """Motorhand's forward chain for one side, from its own assets and retarget config."""
    from motorhand_teleop.retargeting import GloveWujiRetargetPipeline, default_retarget_config

    root = Path(root)
    return GloveWujiRetargetPipeline.from_yaml(
        side, asset_root=root / "assets" / "hand_motor_ve33", config_path=default_retarget_config(root, side)
    )


def apply_pinch(forward, *, d1_cm=None, d2_cm=None, weight=None) -> None:
    """The WujiRetarget's pinch, as the C++ worker's --pinch-*: below d2 (cm, the thumb tip to a finger's tip) that
    tip is pulled to the thumb's with weight (d2 - distance) / (d2 - d1), capped at weight (the package fixes 0.7)."""
    optimizer = forward.retargeter._core.optimizer
    if d1_cm:
        optimizer.d1 = np.full(4, float(d1_cm))
    if d2_cm:
        optimizer.d2 = np.full(4, float(d2_cm))
    if weight:
        cap, tips = float(weight), optimizer.MP_TIP_INDICES

        def pinch_alpha(keypoints):  # AdaptiveOptimizerAnalytical._compute_pinch_alpha, the cap a setting
            distances = np.linalg.norm(keypoints[tips[1:]] - keypoints[tips[0]], axis=1) * 100.0
            alphas = np.clip((optimizer.d2 - distances) / (optimizer.d2 - optimizer.d1 + 1e-8), 0.0, cap)
            return np.concatenate([[np.max(alphas)], alphas])

        optimizer._compute_pinch_alpha = pinch_alpha


def constrain(pose, limits) -> np.ndarray:
    """A retargeted pose as PairRuntime's glove-to-hand2 step sends it: clipped, the four spreads at zero."""
    if pose.diagnostics.get("success") is False:
        raise ValueError("the Wuji retarget did not converge")
    values = np.clip(np.asarray(pose.positions_rad, dtype=np.float64), limits[:, 0], limits[:, 1])
    values[list(SPREAD)] = 0.0
    return values


def target(forward, joints) -> np.ndarray:
    """The 20 targets for one glove pose, solved in this process."""
    return constrain(forward.step(dict(joints)), forward.retargeter.joint_limits)


class SolverError(RuntimeError):
    """The solver failed and is spent: the next frame needs a new one."""


class Solver:
    """The 20 targets for a glove pose: in Motorhand's ForwardRetargetWorker process if own_process and the
    checkout has it, else in this process. mode says which."""

    def __init__(self, root, side: str, *, own_process: bool = True, pinch=None) -> None:
        from motorhand_teleop.retargeting import default_retarget_config

        root = Path(root)
        self._forward = pipeline(root, side)  # in process: the chain; in its own: its joint limits only
        if pinch:  # Motorhand's process builds its own retarget: these settings need this one
            apply_pinch(self._forward, **pinch)
            own_process = False
        self._limits = self._forward.retargeter.joint_limits
        self._process = None
        if own_process:
            try:
                from motorhand_teleop.real_hand.forward_worker import ForwardRetargetWorker
            except ImportError:
                ForwardRetargetWorker = None
            if ForwardRetargetWorker is not None:
                process = ForwardRetargetWorker(hand=side, glove_hand=side,
                                                asset_root=root / "assets" / "hand_motor_ve33",
                                                config_path=default_retarget_config(root, side))
                try:
                    process.start()
                except Exception as error:
                    raise SolverError(f"the retarget process did not start: {describe(error)}") from error
                self._process = process
        self.mode = "own process" if self._process is not None else "in process"

    def solve(self, joints, *, source_ns: int) -> np.ndarray:
        if self._process is None:
            return target(self._forward, joints)
        try:
            pose = self._process.solve(dict(joints), source_ns=source_ns, deadline_ns=None)
        except Exception as error:  # noqa: BLE001 - Motorhand's worker latches on any failure
            raise SolverError(f"the retarget process failed: {describe(error)}") from error
        return constrain(pose, self._limits)

    def close(self) -> None:
        if self._process is not None:
            self._process.close()


class Replies:
    """Each motor's CAN replies, counted on the glove's CAN threads (EncosHand's on_feedback hook)."""

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._channels: dict[int, int] = {}

    def __call__(self, channel: int, motor: int, feedback) -> None:
        self._counts[motor] = self._counts.get(motor, 0) + 1
        self._channels[motor] = channel

    def take(self):
        """The replies since the last take, by motor, and each motor's channel."""
        counts, self._counts = self._counts, {}
        return counts, dict(self._channels)


def glove_hand(port_map, slot_ms: float, replies: Replies | None = None):
    """Motorhand's EncosHand, each CAN thread asking one motor every slot_ms, replies counted."""
    from encos_hand import EncosHand

    hand = EncosHand(port_map=port_map, announce_connection=False)
    for channel in hand.workers.values():
        channel.slot_time_s = slot_ms / 1000.0
        if replies is not None:
            channel.on_feedback = replies
    return hand


def _stats(frames: int, seconds: float, ages, spans, replies: Replies | None, slot_s: float) -> dict:
    stats = {"loop_hz": frames / seconds,
             "age_ms": float(np.median(ages)) if ages else None,
             "span_ms": float(np.median(spans)) if spans else None}
    if replies is not None:
        counts, channels = replies.take()
        if channels:
            per_channel = {channel: sum(1 for c in channels.values() if c == channel) for channel in channels.values()}
            rates = [counts.get(motor, 0) / seconds for motor in channels]
            stats.update(reply_hz=float(np.median(rates)), slowest_reply_hz=float(min(rates)),
                         expected_hz=float(np.median([1.0 / (slot_s * per_channel[c]) for c in channels.values()])))
    return {key: (round(value, 1) if isinstance(value, float) else value) for key, value in stats.items()}


def describe(error: BaseException) -> str:
    parts, seen = [], error
    while seen is not None and len(parts) < 4:
        parts.append(f"{type(seen).__name__}: {seen}")
        seen = seen.__cause__ or seen.__context__
    return " <- ".join(parts)


def run(make_solver, connect_driver, emit, *, sleep=time.sleep, clock=time.monotonic, clock_ns=time.monotonic_ns,
        rate_hz: float = 50.0, retry_s: float = 1.0, slot_s: float = 0.005, replies: Replies | None = None,
        stats_s: float = 10.0, should_stop=lambda: False) -> None:
    """Stream targets from the glove connect_driver() gives, reconnecting with a new one after any failure, solved
    by the solver make_solver() gives (a new one after it fails)."""
    from motorhand_teleop.domain import GLOVE_JOINTS

    period, said, solver = 1.0 / rate_hz, None, None
    try:
        while not should_stop():
            driver, said_by_sdk = connect_driver(), io.StringIO()
            try:
                if solver is None:
                    solver = make_solver()
                with contextlib.redirect_stdout(said_by_sdk):
                    info = driver.connect()
                online = tuple(driver.online_ids)
                missing = [motor for motor in MOTORS if motor not in online]
                if len(missing) == len(MOTORS):
                    raise RuntimeError("no motor online: is the glove's USB-CAN adapter plugged in and readable?")
                if missing:
                    raise RuntimeError(f"motors {', '.join(map(str, missing))} not online (online: {list(online)})")
                emit({"ready": {"online_ids": list(online), "mapping": str(info.get("mapping_source", "")),
                                "slot_ms": round(slot_s * 1000.0, 3), "solver": solver.mode}})
                said, due = None, clock()
                since, frames, ages, spans = due, 0, [], []
                if replies is not None:
                    replies.take()
                while not should_stop():
                    state = driver.read_state()
                    joints = {name: float(state.pose.joints[name]) for name in GLOVE_JOINTS}
                    q = solver.solve(joints, source_ns=int(state.received_ns))
                    emit({"ns": int(state.received_ns), "q": q.tolist()})
                    feedback = getattr(state, "feedback_by_motor", None) or {}
                    stamps = [int(value["received_ns"]) for value in feedback.values() if "received_ns" in value]
                    if stamps:
                        ages.append((clock_ns() - min(stamps)) / 1e6)
                        spans.append((max(stamps) - min(stamps)) / 1e6)
                    frames += 1
                    if stats_s > 0 and clock() - since >= stats_s:
                        emit({"stats": _stats(frames, clock() - since, ages, spans, replies, slot_s)})
                        since, frames, ages, spans = clock(), 0, [], []
                    due += period
                    wait = due - clock()
                    if wait > 0:
                        sleep(wait)
                    else:
                        due = clock()  # behind: carry on from now, no burst
            except Exception as error:  # noqa: BLE001 - said once, then a new connection
                reason = " | ".join([describe(error)] + [line.strip() for line in said_by_sdk.getvalue().splitlines()
                                                         if line.startswith(SDK_REASONS)])
                if reason != said:
                    emit({"status": reason})
                    said = reason
                if isinstance(error, SolverError) and solver is not None:
                    solver.close()
                    solver = None
            finally:
                try:
                    driver.close()
                except Exception:  # noqa: BLE001 - a broken adapter may fail to close too
                    pass
            if not should_stop():
                sleep(retry_s)
    finally:
        if solver is not None:
            solver.close()


class Stop:
    """Asked for by SIGTERM. A plain flag: the handler takes no lock (a threading.Event set there deadlocked about
    once in 300 stops, the signal landing while this thread held the Event's own lock in wait)."""

    def __init__(self) -> None:
        self.requested = False

    def request(self, *_) -> None:
        self.requested = True

    def is_set(self) -> bool:
        return self.requested

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.requested:
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 0.05))


def _die_with_parent() -> None:
    """SIGTERM when the node's process goes, so the glove's serial ports are never left held."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass
    if os.getppid() == 1:
        sys.exit(0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--root", required=True, help="the Motorhand_Teleop checkout")
    parser.add_argument("--rate", type=float, default=50.0, help="glove reads per second")
    parser.add_argument("--ports", default="", help="the glove's USB-CAN adapters, comma separated; "
                        "empty: every CANable2 present")
    parser.add_argument("--mapping", default="", help="the glove's glove_mapping.json; empty: Motorhand's own")
    parser.add_argument("--retry", type=float, default=1.0, help="seconds between connection attempts")
    parser.add_argument("--slot-ms", type=float, default=5.0, help="each CAN thread asks one motor per slot "
                        "(Motorhand's own: 5)")
    parser.add_argument("--stats", type=float, default=10.0, help="seconds between stats lines; 0: none")
    parser.add_argument("--pinch-d1", type=float, default=0.0, help="cm; 0: the retarget yaml's")
    parser.add_argument("--pinch-d2", type=float, default=0.0, help="cm; 0: the retarget yaml's")
    parser.add_argument("--pinch-weight", type=float, default=0.0, help="the pinch weight's cap; 0: the package's 0.7")
    args = parser.parse_args(argv)
    if not args.slot_ms > 0:
        parser.error("--slot-ms must be positive")

    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    _die_with_parent()
    stopping = Stop()
    # The node stops it (SIGTERM); a Ctrl+C in the node's terminal reaches this process too and is left to the node.
    signal.signal(signal.SIGTERM, stopping.request)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    root = Path(args.root)
    ports = [port for port in args.ports.split(",") if port.strip()]
    port_map = {channel: port.strip() for channel, port in enumerate(ports, start=1)} or None

    replies = Replies()

    def connect_driver():
        from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver

        return EncosGloveHardwareDriver(
            hand=args.side, asset_root=root / "assets" / "hand_motor_ve33", config_path=args.mapping or None,
            hand_factory=lambda: glove_hand(port_map, args.slot_ms, replies),
        )

    def emit(message) -> None:
        protocol.write(json.dumps(message) + "\n")

    pinch = {key: value for key, value in (("d1_cm", args.pinch_d1), ("d2_cm", args.pinch_d2),
                                           ("weight", args.pinch_weight)) if value}
    run(lambda: Solver(root, args.side, pinch=pinch), connect_driver, emit, sleep=stopping.sleep, rate_hz=args.rate,
        retry_s=args.retry, slot_s=args.slot_ms / 1000.0, replies=replies, stats_s=args.stats,
        should_stop=stopping.is_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
