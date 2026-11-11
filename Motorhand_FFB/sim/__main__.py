"""Run the glove-only MuJoCo interaction, or plot a recorded session."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from motorhand_teleop.domain import GLOVE_JOINTS
from .device import DemoGlove, HardwareGlove, MAX_FEEDBACK_AGE_NS
from .feedback import FeedbackLimiter, ForceProjector
from .scene import GloveScene
from .telemetry import FINGERS, Recorder, plot_log

# Resume only after the bounded virtual clutch has nearly caught up with input.
_RESUME_ALIGNMENT_RAD = 0.05


def _parser():
    parser = argparse.ArgumentParser(description="Motorhand 手套本体 MuJoCo 交互仿真；默认无硬件、无电机出力")
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="运行仿真并记录 JSONL")
    run.add_argument("--input", choices=("demo", "glove"), default="demo", help="demo 为脚本运动；glove 为真实手套只读输入")
    run.add_argument("--side", choices=("right", "left"), default="right")
    run.add_argument("--mapping", help="手套实际轴映射与标定 JSON")
    run.add_argument("--ports", nargs=3, metavar=("CAN1", "CAN2", "CAN3"),
                     help="三条手套 CAN 适配器串口；真实出力时必须显式指定")
    run.add_argument("--scenario", choices=("press", "grasp"), default="press")
    run.add_argument("--object", choices=("sphere", "box", "cylinder"), default="sphere",
                     help="交互物体：球、方块或圆柱；橡胶/泡棉球为可形变体")
    run.add_argument("--material", choices=("foam", "rubber", "wood", "metal"), default="rubber",
                     help="物体外观和未标定的材料预设；橡胶/泡棉球使用弹性体模型")
    run.add_argument("--object-size", type=float, default=0.05,
                     help="球的直径、方块边长或圆柱直径和高度（米），默认 0.05")
    run.add_argument("--object-motion", choices=("fixed", "free"), default="fixed",
                     help="默认固定中心与整体姿态；free 为带重力和支撑平面的自由物体")
    run.add_argument("--stiffness", choices=("soft", "medium", "hard"),
                     help="可选：覆盖材质的接触或弹性软硬预设")
    run.add_argument("--object-position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="物体中心世界坐标，单位 m")
    run.add_argument("--headless", action="store_true", help="不启动 MuJoCo 窗口")
    charts = run.add_mutually_exclusive_group()
    charts.add_argument("--live-plot", action="store_true", help="无窗口模式也启动本机实时曲线服务")
    charts.add_argument("--no-live-plot", action="store_true", help="不启动实时曲线服务；仍完整记录")
    run.add_argument("--duration", type=float, default=0, help="运行秒数；0 表示直到退出")
    run.add_argument("--rate", type=float, default=50, help="控制循环目标 Hz，并非硬实时保证")
    run.add_argument("--gain", type=float, default=0.1, help="反馈投影增益，范围 0–1")
    run.add_argument("--slew", type=float, default=1.0, help="模型电机名义力矩变化上限 Nm/s")
    run.add_argument("--execute", action="store_true", help="显式启用真实手套电机出力；会先回零")
    run.add_argument("--acknowledge-homing", action="store_true", help="确认真实使能会移动手套回零")
    run.add_argument("--raw-torque-limit-nm", type=float, help="每个原始电机名义软件限额；显式指定 0.01–0.2 Nm")
    run.add_argument("--log", type=Path, help="输出 JSONL；默认 recordings/sim/<时间>.jsonl")
    run.add_argument("--plot", action="store_true", help="停止后生成曲线 PNG")
    plot = sub.add_parser("plot", help="对已记录 JSONL 生成曲线")
    plot.add_argument("log", type=Path)
    plot.add_argument("--output", type=Path, help="输出目录；默认日志旁的同名目录")
    return parser


def _positive(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} 必须为有限正数")
    return value


def _summarize_contacts(contacts):
    forces = {finger: 0.0 for finger in FINGERS}
    depths = {finger: 0.0 for finger in FINGERS}
    tangent_speeds = []
    for contact in contacts:
        finger = contact["finger"]
        normal = float(contact["normal_force_n"])
        depth = float(contact["penetration_m"])
        tangent = float(contact["relative_tangent_speed_m_s"])
        if finger not in forces or not all(math.isfinite(x) and x >= 0 for x in (normal, depth, tangent)):
            raise ValueError("MuJoCo 返回非法接触信息")
        forces[finger] += normal
        depths[finger] = max(depths[finger], depth)
        if normal > 0:
            tangent_speeds.append(tangent)
    active = sum(force > 0.01 for force in forces.values())
    slip = max(tangent_speeds) if tangent_speeds else None
    # An operational indicator, not proof of grasp stability or a slip detector.
    if active >= 2:
        state = "sliding" if slip is not None and slip > 0.01 else "multi_contact"
    else:
        state = "single_contact" if active == 1 else "no_contact"
    return forces, depths, slip, state


def _apply_object_edit(scene, request, *, editing_enabled):
    """Advance a bounded, collision-checked edit without touching real motors."""
    if request is None or not editing_enabled:
        return scene, False, False, True, False
    dimensions = np.asarray(request["dimensions_m"], dtype=float)
    position = np.asarray(request["position_m"], dtype=float)
    quat = np.asarray(request["quat_wxyz"], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("物体编辑位置必须是三个有限数")
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError("物体编辑姿态必须是非零有限四元数")
    quat = quat / np.linalg.norm(quat)
    before_depth = scene.object_max_penetration()
    allowed_depth = max(0.001, before_depth + 1e-5)
    rebuilt = not np.array_equal(dimensions, scene.object_dimensions_m)
    if rebuilt:
        candidate = scene.resized(dimensions)
        if candidate.object_max_penetration() > allowed_depth:
            return scene, False, False, True, True
        scene = candidate
    old_position, old_quat = scene.object_pose()
    if np.dot(old_quat, quat) < 0:
        quat = -quat
    half_angle = math.acos(float(np.clip(np.dot(old_quat, quat), -1, 1)))
    distance = float(np.linalg.norm(position - old_position))
    # Bound the travel of any point, including rotation of an elongated object.
    travel = distance + float(np.linalg.norm(scene.object_dimensions_m)) * half_angle
    fraction = min(1.0, 0.001 / travel) if travel > 1e-12 else 1.0
    if half_angle < 1e-8:
        next_quat = old_quat
    else:
        next_quat = (math.sin((1 - fraction) * half_angle) * old_quat +
                     math.sin(fraction * half_angle) * quat) / math.sin(half_angle)
    moved = travel > 1e-12
    if moved:
        scene.set_object_pose(old_position + fraction * (position - old_position), next_quat)
        if scene.object_max_penetration() > allowed_depth:
            scene.set_object_pose(old_position, old_quat)
            return scene, rebuilt, rebuilt, True, True
    return scene, rebuilt or moved, rebuilt, fraction >= 1.0, False


def _simulation_interval(wall_interval_s, period_s, *, output_enabled):
    """Use fixed simulated time without output; keep the output watchdog strict."""
    if output_enabled:
        if wall_interval_s > 0.1:
            raise TimeoutError(f"控制周期超时: {wall_interval_s * 1000:.1f} ms")
        return wall_interval_s
    return period_s


def run(args):
    _positive(args.rate, "rate")
    _positive(args.slew, "slew")
    if not 20 <= args.rate <= 200:
        raise ValueError("rate 必须在 20–200 Hz 之间")
    if not math.isfinite(args.gain) or not 0 <= args.gain <= 1:
        raise ValueError("gain 必须在 0–1 之间")
    if not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError("duration 必须是非负有限秒数")
    if args.execute and args.input != "glove":
        raise ValueError("--execute 需要 --input glove")
    if args.headless and args.duration == 0:
        raise ValueError("--headless 需要指定正数 --duration，避免无界运行")

    source = (DemoGlove(args.side, args.scenario) if args.input == "demo" else
              HardwareGlove(args.side, args.mapping, args.execute, args.acknowledge_homing,
                            args.raw_torque_limit_nm, args.ports))
    recorder = None
    viewer = None
    live = None
    started = time.monotonic()
    period = 1.0 / args.rate
    log_path = args.log or Path("recordings/sim") / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
    try:
        if args.input == "glove":
            source.connect()
        first = source.read()
        scene = GloveScene(args.side, args.scenario, args.stiffness,
                           object_shape=args.object, object_material=args.material,
                           object_size_m=args.object_size, object_fixed=args.object_motion == "fixed",
                           object_position_m=args.object_position, glove_model=source.model,
                           initial_tree_positions_rad=first.tree_positions_rad)
        projector = ForceProjector(source.model, scene.model)
        limits = source.torque_limits_nm if args.execute else np.full(12, 0.2)
        limiter = FeedbackLimiter(limits, np.full(12, args.slew))
        metadata = {
            "schema": "motorhand-sim-3", "mujoco_version": __import__("mujoco").__version__,
            "input": args.input, "execute": args.execute, "gain": args.gain,
            "target_rate_hz": args.rate, "slew_nm_s": args.slew,
            "raw_torque_limit_nm": args.raw_torque_limit_nm if args.execute else None,
            "model_torque_limit_nm": limits.tolist(),
            "motor_names": list(source.model.motor_names), "joint_names": list(GLOVE_JOINTS),
            "scene": scene.metadata,
            "clock_domains": {k: "monotonic" for k in ("time_ns", "input_time_ns", "contact_time_ns", "command_time_ns")},
            "sources": {"motor_position_rad": "measured" if args.input == "glove" else "simulated",
                        "joint_position_rad": "estimated" if args.input == "glove" else "simulated",
                        "input_time_ns": "measured_hardware_timestamp" if args.input == "glove" else "software_clock",
                        "motor_current_a": "measured" if args.input == "glove" else "unavailable",
                        "computed_torque_nm": "derived_from_simulated_contact", "command_torque_nm": "command"},
            "notes": ["CAD mesh convex-hull contact is not a human fingertip",
                      "contact forces and their virtual-work projection use the simulated glove pose, not the input pose",
                      "input and simulated glove positions may differ while contact blocks motion",
                      "command torque is nominal, not a calibrated measured torque",
                      "displacement is maximum per-frame simulated contact penetration",
                      "object_deformation_m is the maximum radial vertex change relative to the centroid",
                      "deformable object position and orientation are vertex centroid and best-fit material frame",
                      "resizing recompiles the object and resets its deformation and velocity",
                      "during object dragging zero-current mode is requested; hardware zero force is not verified",
                      "force-mode resume sends a near-zero initialization frame; periodic commands are logged separately",
                      "grasp_state is a multi-contact/slip heuristic, not verified grasp stability",
                      "latency timestamps describe software stages, not physical motor response"],
        }
        recorder = Recorder(log_path, metadata)
        if not args.headless:
            from .viewer import LiveViewer
            viewer = LiveViewer(scene, editing_enabled=not args.execute)
        if args.live_plot or (not args.headless and not args.no_live_plot):
            from .live_plot import LivePlot
            live = LivePlot(source.model.motor_names, scene.metadata)
            print(f"实时曲线（仅本机）：{live.url}", flush=True)
        if args.execute:
            print("警告：即将真实回零并使能手套；出现异常会立即退出并失能。", file=sys.stderr, flush=True)
            source.enable_output()
        if args.input == "demo":
            source.reset_clock()
        startup_sample = source.read()
        scene.synchronize_tree(startup_sample.tree_positions_rad)
        if args.execute:
            initial_frame = scene.step(startup_sample.tree_positions_rad, period)
            if any(c["normal_force_n"] > 0.01 or c["penetration_m"] > 1e-6
                   for c in initial_frame["contacts"]):
                raise RuntimeError("回零后的手套已与物体接触；已停止，请调整物体位置后重启")
        print(f"物体: {args.object}；尺寸 {args.object_size:.3f} m；"
              f"材质 {args.material}；接触 {scene.stiffness}；"
              f"运动 {args.object_motion}", flush=True)
        print(f"记录: {log_path}; 输入={args.input}; 实际出力={args.execute}", flush=True)
        started = time.monotonic()
        last_sample_time = None
        frames = 0
        peak_force = 0.0
        pending_edit = None
        feedback_paused = False
        object_dragging = False
        while args.duration == 0 or time.monotonic() - started < args.duration:
            cycle_start = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            reset_object = viewer is not None and viewer.consume_reset()
            if reset_object:
                if args.execute:
                    print("力控输出期间禁止重置物体；请先停止仿真。", file=sys.stderr, flush=True)
                    reset_object = False
                else:
                    scene.reset_object()
                    pending_edit = None
                    limiter.reset()
            feedback_transition = None
            # Drain ordered press/release events before deciding to resume.
            while viewer is not None:
                request = viewer.consume_edit()
                if request is None:
                    break
                if args.execute:
                    if not isinstance(request.get("dragging"), bool):
                        raise ValueError("真实力控的物体平移请求必须包含拖动状态")
                    if not feedback_paused:
                        # Stop the owned output before touching the physical scene.
                        source.pause_output()
                        feedback_paused = True
                        feedback_transition = "pause_requested"
                        last_sample_time = None
                        limiter.reset()
                        print("已请求暂停手套力输出；请松开手指并等待仿真姿态对齐。"
                              "如电机仍出力，请立即使用物理急停。", flush=True)
                    object_dragging = request["dragging"]
                pending_edit = request
            if args.execute and pending_edit is not None:
                # Only translation is authorized while real output is enabled.
                _, current_quat = scene.object_pose()
                pending_edit = {"position_m": pending_edit["position_m"],
                                "quat_wxyz": current_quat,
                                "dimensions_m": scene.object_dimensions_m.copy()}
            scene, object_edited, rebuilt, edit_done, edit_blocked = _apply_object_edit(
                scene, pending_edit, editing_enabled=not args.execute or feedback_paused
            )
            if edit_done:
                pending_edit = None
            if edit_blocked and viewer is not None:
                viewer.notify_blocked()
            if object_edited:
                limiter.reset()
            if rebuilt:
                projector = ForceProjector(source.model, scene.model)
            sample = source.read()
            if args.input == "glove":
                age = time.monotonic_ns() - sample.received_ns
                if not 0 <= age < MAX_FEEDBACK_AGE_NS:
                    raise TimeoutError(f"手套输入已过期: {age / 1e6:.1f} ms")
            now_sample = time.monotonic()
            wall_interval = period if last_sample_time is None else now_sample - last_sample_time
            last_sample_time = now_sample
            dt = _simulation_interval(wall_interval, period,
                                      output_enabled=args.execute and not feedback_paused)
            frame = scene.step(sample.tree_positions_rad, dt, edit_clutch=feedback_paused)
            contact_time_ns = time.monotonic_ns()
            contacts = frame["contacts"]
            forces, depths, slip, grasp = _summarize_contacts(contacts)
            if contacts:
                projected = projector.project(scene.data, contacts, frame["virtual_motor_positions_rad"],
                                              frame["virtual_tree_positions_rad"])
                requested = projected * args.gain
                if feedback_paused:
                    limiter.reset()
                    computed = np.zeros(12)
                else:
                    computed = limiter.apply(requested, dt)
                    if limiter.last_error:
                        raise RuntimeError(f"反馈限制器失效: {limiter.last_error}")
            else:
                limiter.reset()
                requested = computed = np.zeros(12)
            pause_reason = (("dragging" if object_dragging else "waiting_clear")
                            if feedback_paused else None)
            if viewer is not None and feedback_paused:
                viewer.set_feedback_state(pause_reason)
            command, command_time_ns = None, None
            if args.execute and not feedback_paused:
                command = source.send(computed, sample.received_ns)
                command_time_ns = time.monotonic_ns()
            elif args.execute and not object_dragging and pending_edit is None:
                if (frame["input_alignment_error_rad"] <= _RESUME_ALIGNMENT_RAD
                        and not any(c["normal_force_n"] > 0.01 or c["penetration_m"] > 1e-6
                                    for c in contacts)):
                    # Resuming the already-homed session must never home again.
                    source.enable_output()
                    feedback_paused = False
                    feedback_transition = "resume_requested"
                    last_sample_time = None
                    print("物体到位、仿真接触低于恢复阈值且姿态已对齐；已请求恢复力反馈。", flush=True)
                    if viewer is not None:
                        viewer.set_feedback_state("active")
            _, object_quat = scene.object_pose()
            record = {
                "time_ns": time.monotonic_ns(), "sim_time_s": frame["sim_time_s"],
                "control_interval_s": wall_interval, "simulation_step_s": dt,
                "input_time_ns": sample.received_ns, "contact_time_ns": contact_time_ns,
                "command_time_ns": command_time_ns, "contact_force_n": forces,
                "contact_penetration_m": depths, "displacement_m": max(depths.values()),
                "contacts": contacts, "computed_torque_nm": computed.tolist(),
                "requested_torque_nm": requested.tolist(),
                "torque_limited": (np.abs(requested) > limits).tolist(),
                "slew_limited": ((np.abs(computed - np.clip(requested, -limits, limits)) > 1e-12).tolist()
                                 if pause_reason is None else [False] * 12),
                "command_torque_nm": command,
                "motor_position_rad": [sample.motor_positions_rad[name] for name in source.model.motor_names],
                "joint_position_rad": [sample.joint_positions_rad[name] for name in GLOVE_JOINTS],
                "sim_motor_position_rad": [frame["virtual_motor_positions_rad"][name] for name in source.model.motor_names],
                "sim_joint_position_rad": [frame["virtual_joint_positions_rad"][name] for name in GLOVE_JOINTS],
                "virtual_servo_mode": frame["virtual_servo_mode"],
                "virtual_servo_limit_nm": frame["virtual_servo_limit_nm"],
                "input_alignment_error_rad": frame["input_alignment_error_rad"],
                "motor_current_a": sample.motor_current_a,
                "object_position_m": frame["object_position_m"].tolist(),
                "object_deformation_m": frame["object_deformation_m"],
                "object_velocity_m_s": frame["object_velocity_m_s"].tolist(),
                "object_orientation_wxyz": object_quat.tolist(),
                "object_dimensions_m": scene.object_dimensions_m.tolist(),
                "object_edited": object_edited,
                "object_edit_blocked": edit_blocked,
                "feedback_pause_reason": pause_reason,
                "feedback_transition": feedback_transition,
                "grasp_state": grasp, "slip_speed_m_s": slip,
                "object_reset": bool(reset_object),
            }
            recorder.append(record)
            if live is not None:
                live.append({key: value for key, value in record.items() if key != "contacts"})
            frames += 1
            peak_force = max(peak_force, *forces.values())
            if viewer is not None:
                viewer.submit(scene)
            remaining = period - (time.monotonic() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)
        print(json.dumps({"frames": frames, "peak_simulated_finger_force_n": peak_force,
                          "hardware_command_enabled": args.execute, "log": str(log_path)}, ensure_ascii=False), flush=True)
    finally:
        # Disable the physical output before potentially waiting on UI or disk.
        try:
            source.close()
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                try:
                    if live is not None:
                        live.close()
                finally:
                    if recorder is not None:
                        recorder.close()
    if args.plot:
        output = log_path.with_suffix("")
        print("曲线:", *(str(p) for p in plot_log(log_path, output)), sep="\n")


def main():
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.action == "plot":
            print("\n".join(str(path) for path in plot_log(args.log, args.output or args.log.with_suffix(""))))
        else:
            run(args)
    except KeyboardInterrupt:
        print("已停止；设备已执行关闭流程。", file=sys.stderr)
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"仿真已停止：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
