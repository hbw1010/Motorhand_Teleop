"""Motorhand exoskeleton gloves -> Wuji Hand 2 joint commands, in place of the MANUS glove node and retarget.

Each glove has a worker running Motorhand's forward chain: by default the C++ one (native/, built from Motorhand's
own C++ by bash/build_motorhand_glove.sh: ~0.5 ms a frame, no GIL against the CAN threads), else worker.py in the
Motorhand_Teleop checkout's own Pixi env; both speak the same JSON lines. This node sends each target exactly as
wuji_hand2's retarget does: JointState with
COMMAND_JOINT_NAMES on /<hand>/joint_commands, best effort, the hand names from the left_hand/right_hand
parameters (launch/hands/hand2_nodes.py sets them as for retarget: the drivers, or /<prefix>_<side>). A worker
that exits is started again after restart_s.

With force_feedback or follow_hand the node also tells the worker, on its stdin, the Wuji hand's state
(/<side>_hand/joint_states, in the command joints' order) and its final target (/<side>_hand/joint_commands) with
whether that target is one this glove sent (in the last 0.5 s). The C++ worker then homes the glove and drives it as
Motorhand's own session does (native/glove_control.hpp): force feedback while the hand follows this glove, the glove
moved to the hand's pose while it follows something else (DAgger's model).

Parameters (config/motorhand_glove.yaml):
  root             the Motorhand_Teleop checkout, `pixi install --frozen` done; empty: beside this workspace
  gloves           the hands that have a glove
  rate_hz          glove reads per second
  slot_ms          each CAN thread asks one motor per slot (Motorhand's own: 5 ms; a motor is read every slot x
                   the motors on its adapter)
  stats_s          seconds between each glove's stats line in the log (0: none)
  <side>_ports     that glove's USB-CAN adapters, comma separated; empty: every CANable2 present (one glove only)
  <side>_mapping   that glove's glove_mapping.json; empty: Motorhand's own
  restart_s        the pause before a worker that exited is started again
  worker           native (the C++ worker) or python
  native_binary    the C++ worker; empty: build/motorhand-glove-native/motorhand_glove_native in this workspace
  pinch_d1_cm, pinch_d2_cm, pinch_weight   the WujiRetarget's pinch (0: Motorhand's own, 2 and 4 cm, 0.7): below d2
                   a finger tip is pulled to the thumb's with weight (d2 - distance) / (d2 - d1), capped at the weight
  force_feedback   while the hand follows this glove, the glove's motors push back where the hand is blocked
  follow_hand      while the hand follows something else, the glove's motors move it to the hand's pose
  follow_kp        how stiffly (MIT kp, Nm/rad, at most Motorhand's 45): a finger held away from the hand's pose
                   is pushed with kp x the angle, all the time it is held there
  home_glove       with either: home the glove when its worker starts (Motorhand's driver drives nothing before;
                   its motors take it to the zero pose at 5 rpm); false: its output starts where it is
  <side>_feedback_limits   that hand's limits config; empty: config/motorhand_feedback/max-torque-linear-<side>.json
  <side>_feedback_contact  the contact-impedance config; empty: config/motorhand_feedback/contact-impedance-<side>.json
                   if there is one, else Motorhand's own for that side
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from wuji_hand2.joints import COMMAND_JOINT_NAMES, NUM_JOINTS, joint_index

SIDES = ("left", "right")
WORKER = Path(__file__).resolve().with_name("worker.py")
OURS_S = 0.5  # how far back a target sent by this glove still counts as its own
OURS_TOLERANCE = 1e-6  # rad: DAgger's hand bridge passes a glove target through as o + 1 * (t - o)
# Not for the worker: this env's Python packages and libraries (the Motorhand env has its own ABI).
FOREIGN = ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "AMENT_PREFIX_PATH",
           "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH")


class Supervisor:
    """Runs command, gives each stdout line to on_line and the exit code to on_exit, and starts it again
    restart_s after it exits, until stop()."""

    def __init__(self, command, on_line, on_exit, *, restart_s: float = 2.0, environment=None,
                 feed: bool = False) -> None:
        self._command, self._on_line, self._on_exit = list(command), on_line, on_exit
        self._restart_s, self._environment, self._feed = restart_s, environment, feed
        self._lock, self._stopping = threading.Lock(), threading.Event()
        self.process = None
        self._thread = threading.Thread(target=self._run, name="motorhand-glove-worker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._stopping.is_set():
                    return
                try:
                    self.process = subprocess.Popen(
                        self._command, stdin=subprocess.PIPE if self._feed else subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=None,
                        env=self._environment, text=True, bufsize=1,
                    )
                    process = self.process
                except OSError as error:
                    process = None
                    code = str(error)
            if process is not None:
                for line in process.stdout:
                    self._on_line(line.rstrip("\n"))
                code = process.wait()
            if self._stopping.is_set():
                return
            self._on_exit(code)
            self._stopping.wait(self._restart_s)

    def send(self, line: str) -> None:
        """A line to the worker's stdin (with feed); dropped while it is not running."""
        process = self.process
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.write(line)
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def stop(self, timeout_s: float = 3.0) -> None:
        with self._lock:
            self._stopping.set()
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._thread.join(timeout_s)


@dataclass
class Glove:
    hand: str
    publisher: object
    supervisor: Supervisor | None = None
    streaming: bool = False
    sent: deque = field(default_factory=lambda: deque(maxlen=200))  # (monotonic ns, the 20) this glove sent
    # sent grows on the worker's reader thread and is read on the ROS executor's: iterating it while it grows raises
    lock: threading.Lock = field(default_factory=threading.Lock)


class MotorhandGloveNode(Node):
    def __init__(self, *, worker_command=None, **kwargs) -> None:
        super().__init__("motorhand_glove", **kwargs)
        hands = {side: str(self.declare_parameter(f"{side}_hand", f"{side}_hand").value) for side in SIDES}
        gloves = [str(side) for side in self.declare_parameter("gloves", ["right"]).value]
        root = str(self.declare_parameter("root", "").value)
        rate_hz = float(self.declare_parameter("rate_hz", 100.0).value)
        restart_s = float(self.declare_parameter("restart_s", 2.0).value)
        slot_ms = float(self.declare_parameter("slot_ms", 5.0).value)
        stats_s = float(self.declare_parameter("stats_s", 10.0).value)
        if not slot_ms > 0:
            raise RuntimeError(f"slot_ms:={slot_ms:g} must be positive")
        ports = {side: str(self.declare_parameter(f"{side}_ports", "").value) for side in SIDES}
        mappings = {side: str(self.declare_parameter(f"{side}_mapping", "").value) for side in SIDES}
        kind = str(self.declare_parameter("worker", "native").value)
        pinch = [float(self.declare_parameter(name, 0.0).value) for name in ("pinch_d1_cm", "pinch_d2_cm", "pinch_weight")]
        native = str(self.declare_parameter("native_binary", "").value)
        force = bool(self.declare_parameter("force_feedback", False).value)
        follow = bool(self.declare_parameter("follow_hand", False).value)
        follow_kp = float(self.declare_parameter("follow_kp", 15.0).value)
        home = bool(self.declare_parameter("home_glove", True).value)
        if kind == "python" and (force or follow):
            self.get_logger().warning("force_feedback and follow_hand need the C++ worker: off with worker:=python")
            force = follow = False
        self._feedback = force or follow  # the worker is told the hand's state
        limits = {side: str(self.declare_parameter(f"{side}_feedback_limits", "").value) for side in SIDES}
        contacts = {side: str(self.declare_parameter(f"{side}_feedback_contact", "").value) for side in SIDES}
        if kind not in ("native", "python"):
            raise RuntimeError(f"worker:={kind!r}: native or python")
        unknown = sorted(set(gloves) - set(SIDES))
        if unknown:
            raise RuntimeError(f"gloves: {unknown} is not a hand (left, right)")
        driven = [side for side in SIDES if side in gloves and hands[side]]
        if len(driven) > 1 and not all(ports[side] for side in driven):
            raise RuntimeError("with two gloves give each its <side>_ports: found by type, the adapters of one "
                               "glove cannot be told from the other's")

        root_path = Path(root) if root else _beside_workspace()
        env_root = root_path / ".pixi" / "envs" / "default"
        environment = {name: value for name, value in os.environ.items() if name not in FOREIGN}
        environment.update(PATH=f"{env_root / 'bin'}{os.pathsep}{environment.get('PATH', '')}",
                           CONDA_PREFIX=str(env_root), PYTHONNOUSERSITE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")  # Motorhand's activation
        if worker_command is None:
            if kind == "native":
                program = [native or str(_workspace() / "build" / "motorhand-glove-native" / "motorhand_glove_native")]
                if not Path(program[0]).is_file():
                    raise RuntimeError(f"no C++ glove worker ({program[0]}): bash bash/build_motorhand_glove.sh, "
                                       "or worker:=python")
            else:
                python = env_root / "bin" / "python"
                if not python.is_file():
                    raise RuntimeError(f"no Motorhand env ({python}): run `pixi install --frozen` in {root_path}")
                program = [str(python), "-I", "-u", str(WORKER)]
            serials = _hand_serials() if force else {}

            def worker_command(side):
                command = [*program, "--side", side, "--root", str(root_path), "--rate", str(rate_hz),
                           "--slot-ms", str(slot_ms), "--stats", str(stats_s), "--ports", ports[side],
                           "--mapping", mappings[side], "--pinch-d1", str(pinch[0]), "--pinch-d2", str(pinch[1])]
                if pinch[2]:
                    command += ["--pinch-weight", str(pinch[2])]
                if force:
                    ours = _workspace() / "config" / "motorhand_feedback"
                    contact = contacts[side] or (str(ours / f"contact-impedance-{side}.json")
                                                 if (ours / f"contact-impedance-{side}.json").is_file() else "")
                    command += ["--feedback-limits", limits[side] or str(ours / f"max-torque-linear-{side}.json"),
                                "--feedback-contact", contact, "--wuji-serial", serials.get(side, "")]
                if follow:
                    command += ["--follow", "--follow-kp", str(follow_kp)]
                if (force or follow) and not home:
                    command += ["--no-home"]
                return command

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._restart_s = restart_s
        self._gloves: dict[str, Glove] = {}
        for side in driven:
            glove = Glove(hands[side], self.create_publisher(JointState, f"/{hands[side]}/joint_commands", qos))
            self._gloves[side] = glove
            glove.supervisor = Supervisor(
                worker_command(side), lambda line, side=side: self._on_line(side, line),
                lambda code, side=side: self._on_exit(side, code), restart_s=restart_s, environment=environment,
                feed=self._feedback,
            )
            if self._feedback:
                self.create_subscription(JointState, f"/{side}_hand/joint_states",
                                         lambda msg, side=side: self._on_hand_state(side, msg), qos)
                self.create_subscription(JointState, f"/{side}_hand/joint_commands",
                                         lambda msg, side=side: self._on_hand_command(side, msg), qos)
        self.get_logger().info(
            f"Motorhand gloves ({root_path}): "
            + (", ".join(f"{side} -> /{glove.hand}/joint_commands" for side, glove in self._gloves.items())
               or "none (no hand has both a glove and a topic)")
        )

    def _on_line(self, side: str, line: str) -> None:
        glove, log = self._gloves.get(side), self.get_logger()
        if glove is None:
            return
        try:
            message = json.loads(line)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            log.info(f"{side} glove: {line}")
            return
        if "q" in message:
            values = message["q"]
            if not (isinstance(values, list) and len(values) == NUM_JOINTS
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)):
                log.warning(f"{side} glove: dropped a bad frame", throttle_duration_sec=2.0)
                return
            command = JointState()
            command.header.stamp = self.get_clock().now().to_msg()
            command.header.frame_id = glove.hand
            command.name = list(COMMAND_JOINT_NAMES)
            command.position = [float(v) for v in values]
            glove.publisher.publish(command)
            with glove.lock:
                glove.sent.append((time.monotonic_ns(), command.position))
            if not glove.streaming:
                glove.streaming = True
                log.info(f"{side} glove: streaming to /{glove.hand}/joint_commands")
        elif "stats" in message:
            log.info(f"{side} glove: {_stats_text(message['stats'])}")
        elif "ready" in message:
            log.info(f"{side} glove connected: {message['ready']}")
        elif "event" in message:
            log.info(f"{side} {message['event']}")
        elif "status" in message:
            glove.streaming = False
            log.warning(f"{side} glove not streaming: {message['status']}")

    def _ordered(self, side: str, msg, *fields):
        """The message's fields in the command joints' order (matched by name), or None if it lacks a joint."""
        values = [[0.0] * NUM_JOINTS for _ in fields]
        seen = set()
        for k, name in enumerate(msg.name):
            index = joint_index(name, side)
            if index is None:
                continue
            for out, source in zip(values, fields):
                if k >= len(source):
                    return None
                out[index] = float(source[k])
            seen.add(index)
        return values if len(seen) == NUM_JOINTS else None

    def _on_hand_state(self, side: str, msg) -> None:
        glove = self._gloves.get(side)
        if not self._feedback or glove is None or glove.supervisor is None:
            return
        ordered = self._ordered(side, msg, msg.position, msg.velocity, msg.effort)
        if ordered is not None:
            q, dq, effort = ordered
            glove.supervisor.send(json.dumps({"hand": {"ns": time.monotonic_ns(), "q": q, "dq": dq,
                                                       "effort": effort}}) + "\n")

    def _on_hand_command(self, side: str, msg) -> None:
        glove = self._gloves.get(side)
        if not self._feedback or glove is None or glove.supervisor is None:
            return
        ordered = self._ordered(side, msg, msg.position)
        if ordered is None:
            return
        target, now = ordered[0], time.monotonic_ns()
        with glove.lock:
            frames = list(glove.sent)
        ours = any(now - sent_ns <= OURS_S * 1e9 and
                   all(abs(a - b) <= OURS_TOLERANCE for a, b in zip(target, sent))
                   for sent_ns, sent in reversed(frames))
        glove.supervisor.send(json.dumps({"target": {"ns": now, "q": target, "ours": ours}}) + "\n")

    def _on_exit(self, side: str, code) -> None:
        self._gloves[side].streaming = False
        self.get_logger().error(f"{side} glove worker ended ({code}); starting it again in {self._restart_s:g} s")

    def destroy_node(self) -> None:
        for glove in self._gloves.values():
            if glove.supervisor is not None:
                glove.supervisor.stop()
        super().destroy_node()


def _stats_text(stats: dict) -> str:
    text = f"{stats.get('loop_hz')} Hz out"
    if "motor_hz" in stats:  # the C++ worker
        text += (f"; motors read {stats['motor_hz']} Hz (slowest {stats.get('slowest_motor_hz')}); "
                 f"solve {stats.get('solve_ms')} ms")
    if "pinch_cm" in stats:
        text += "; closest to the thumb (cm) " + " ".join(
            f"{finger} {cm}" for finger, cm in zip(("index", "middle", "ring", "pinky"), stats["pinch_cm"]))
    if "reply_hz" in stats:
        text += (f"; motors answer {stats['reply_hz']} Hz of {stats['expected_hz']} the slot allows"
                 f" (slowest {stats['slowest_reply_hz']})")
    text += f"; a frame's readings {stats.get('age_ms')} ms old when sent, {stats.get('span_ms')} ms apart"
    glove = stats.get("glove")
    if glove:  # the glove's output (the C++ worker's glove_control)
        text += f"; glove {glove.get('mode')}"
        if "contact_pct" in glove and glove.get("mode") == "force":
            text += (f": contact {glove['contact_pct']}% of the time, push up to {glove.get('peak_nm')} Nm"
                     f" ({glove.get('peak_axis') or '-'}), Wuji current up to {glove.get('max_current_a')} A,"
                     f" error up to {glove.get('max_error_rad')} rad")
        if "follow_waiting_pct" in glove and glove.get("mode") == "follow":
            text += f": waiting for the inverse {glove['follow_waiting_pct']}%, failed {glove.get('inverse_failed')}"
    return text


def _workspace() -> Path:
    from tianji_runtime.resources import workspace

    return workspace()


def _beside_workspace() -> Path:
    return _workspace().parent / "Motorhand_Teleop"


def _hand_serials() -> dict:
    """Each hand's serial (config/robot.json), for the force feedback's limits config to be checked against."""
    robot = json.loads((_workspace() / "config" / "robot.json").read_text())
    return {side: str(robot[f"{side}_hand"]["serial"]) for side in SIDES if f"{side}_hand" in robot}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = MotorhandGloveNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A second Ctrl+C (the terminal's and launch's both arrive) must not cut the workers' stop short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
