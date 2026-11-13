"""Encos 12轴电机手套的硬件适配器。

该模块将SDK原始电机degree/Nm与模型电机rad及真实指节GlovePose明确分开。
电机ID到模型电机的绑定、零位和轴比例须经实机标定；未标定配置禁止控制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from ..domain import GLOVE_JOINTS, GlovePose
from ..ve33_adapter import Ve33GloveModel
from ..ve33_kinematics import Ve33Pose


class EncosHardwareUnavailable(RuntimeError):
    """Encos SDK、串口/CAN 通道或节点状态不可用。"""


class EncosCalibrationRequired(RuntimeError):
    """电机轴序、零位或力矩常数尚未完成标定。"""


@dataclass(frozen=True)
class EncosMotorConfig:
    motor_id: int
    joint: str
    zero_deg: float
    sign: float
    scale: float
    min_deg: float
    max_deg: float
    speed_rpm: float
    current_limit_a: float

    def __post_init__(self) -> None:
        if self.motor_id <= 0:
            raise ValueError("motor_id必须为正整数")
        if self.joint not in GLOVE_JOINTS:
            raise ValueError(f"未知手套独立坐标: {self.joint}")
        values = (
            self.zero_deg,
            self.sign,
            self.scale,
            self.min_deg,
            self.max_deg,
            self.speed_rpm,
            self.current_limit_a,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"motor {self.motor_id} 标定参数必须有限")
        if self.sign == 0.0 or self.scale == 0.0:
            raise ValueError(f"motor {self.motor_id} sign/scale不能为0")
        if self.min_deg >= self.max_deg:
            raise ValueError(f"motor {self.motor_id} 软限位非法")
        if self.speed_rpm < 0.0 or self.current_limit_a < 0.0:
            raise ValueError(f"motor {self.motor_id} 速度/电流上限不能为负")


@dataclass(frozen=True)
class EncosGloveState:
    """真实指节姿态，以及明确分列的SDK原始角/模型电机角。"""

    pose: GlovePose
    feedback_by_motor: Mapping[int, Mapping[str, object]]
    online_ids: tuple[int, ...]
    received_ns: int
    raw_motor_positions_deg: Mapping[int, float] = field(default_factory=dict)
    motor_positions_rad: Mapping[str, float] = field(default_factory=dict)
    tree_positions_rad: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("raw_motor_positions_deg", "motor_positions_rad", "tree_positions_rad"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class EncosGloveHardwareDriver:
    """把 EncosHand 接到项目的 GloveDriver 接口。

    位置命令的单位在 seam 内统一为 rad；Encos SDK 仍接收 degree。电流模式
    必须提供真实Kt；显式force_torque_limits_nm模式直接发送受限名义Nm，不代表实测力矩。
    """

    joint_names = GLOVE_JOINTS
    motor_ids = tuple(range(1, 13))

    feedback_timeout_ns = 100_000_000
    home_current_a = MappingProxyType({1: 1.5, **dict.fromkeys(range(2, 13), 1.0)})
    def __init__(
        self,
        *,
        hand: str = "right",
        asset_root: str | Path | None = None,
        hand_factory: Callable[[], Any] | None = None,
        config_path: str | Path | None = None,
        mapping: Mapping[int, Mapping[str, object]] | None = None,
        allow_uncalibrated: bool = False,
        torque_constant_nm_per_a: float | Mapping[int | str, float] | None = None,
        force_torque_limits_nm: Mapping[int | str, float] | None = None,
        max_haptic_current_a: float | None = None,
        haptic_kp: float | None = None,
        haptic_kd: float | None = None,
    ) -> None:
        if hand not in ("left", "right"):
            raise ValueError("hand必须为left或right")
        self.hand = hand
        self.asset_root = Path(asset_root) if asset_root is not None else None
        self._model: Ve33GloveModel | None = None
        self.config_path = Path(config_path) if config_path else _default_mapping_path()
        raw_config = _mapping_config(mapping, self.config_path)
        self.mapping_calibrated = bool(raw_config.get("calibrated", False))
        self.mapping_source = str(raw_config.get("mapping_source", "unknown"))
        self.motors = _parse_motor_configs(raw_config.get("motors"))
        haptic = raw_config.get("haptic", {})
        if not isinstance(haptic, Mapping):
            raise ValueError("haptic配置必须是对象")
        configured_torque = haptic.get("torque_constant_nm_per_a")
        torque_calibration = (
            torque_constant_nm_per_a
            if torque_constant_nm_per_a is not None
            else configured_torque
        )
        self._torque_constants_by_id = _normalize_torque_constants(torque_calibration)
        self.torque_constant_nm_per_a = (
            self._torque_constants_by_id
            if isinstance(torque_calibration, Mapping)
            else None if torque_calibration is None else float(torque_calibration)
        )
        self._force_torque_limits_by_id = _normalize_force_torque_limits(force_torque_limits_nm)
        self.max_haptic_current_a = float(
            max_haptic_current_a
            if max_haptic_current_a is not None
            else haptic.get("max_current_a", 0.25)
        )
        self.haptic_kp = float(haptic_kp if haptic_kp is not None else haptic.get("kp", 45.0))
        self.haptic_kd = float(haptic_kd if haptic_kd is not None else haptic.get("kd", 0.5))
        if not np.isclose((self.haptic_kp, self.haptic_kd), (45.0, 0.5), rtol=0.0, atol=1e-12).all():
            raise ValueError("Encos实时MIT参数固定为kp=45.0、kd=0.5")
        if not np.all(np.isfinite((self.max_haptic_current_a, self.haptic_kp, self.haptic_kd))):
            raise ValueError("触觉参数必须有限")
        if self.max_haptic_current_a < 0.0 or self.haptic_kp < 0.0 or self.haptic_kd < 0.0:
            raise ValueError("触觉参数不能为负")

        self._hand_factory = hand_factory or _default_hand_factory
        self._hand = None
        self._connected = False
        self._closed = False
        self._online_ids: tuple[int, ...] = ()
        self._owned_output = False
        self._homed = False
        self._output_enabled = False
        self._force_enabled = False
        self._feedback_seed = None
        self._target_seed = None
        self._position_cache = None

        if not allow_uncalibrated and not self.mapping_calibrated:
            self._require_calibration_on_send = True
        else:
            self._require_calibration_on_send = False

    @property
    def connected(self) -> bool:
        return self._connected and not self._closed

    @property
    def active(self) -> bool:
        return self.connected and bool(self._online_ids)

    @property
    def online_ids(self) -> tuple[int, ...]:
        return self._online_ids

    @property
    def model_motor_limits_rad(self) -> Mapping[str, tuple[float, float]]:
        """现有raw软件限位按零位／方向／比例转换成真实模型电机边界，不近似指节。"""
        model = self._glove_model()
        limits = {}
        for config in self.motors.values():
            edges = np.deg2rad(np.asarray((config.min_deg, config.max_deg)) - config.zero_deg)
            edges *= config.sign * config.scale
            limits[model.joint_motor_names[config.joint]] = (
                float(np.min(edges)), float(np.max(edges)),
            )
        return MappingProxyType(limits)

    @property
    def motor_names(self) -> tuple[str, ...]:
        return self._glove_model().motor_names

    @property
    def model_motor_torque_constants_nm_per_a(self) -> Mapping[str, float]:
        """模型电流与模型电机力矩共轭；轴方向／比例在SDK边界单独转换一次。"""
        if self._torque_constants_by_id is None:
            raise EncosCalibrationRequired("缺少Encos电机 torque_constant_nm_per_a 标定")
        model = self._glove_model()
        return MappingProxyType({
            model.joint_motor_names[config.joint]: self._torque_constants_by_id[mid]
            for mid, config in self.motors.items()
        })

    @property
    def model_current_limits_a(self) -> Mapping[str, float]:
        """raw电流保护除以轴比例绝对值，返回完整模型电机轴cap。"""
        model = self._glove_model()
        return MappingProxyType({
            model.joint_motor_names[config.joint]: min(
                self.max_haptic_current_a, config.current_limit_a,
            ) / abs(config.sign * config.scale)
            for config in self.motors.values()
        })

    @property
    def model_force_torque_limits_nm(self) -> Mapping[str, float]:
        """可发送连续名义Nm上限：raw软件cap减半LSB裕量，再除轴比例绝对值。"""
        limits = self._require_force_torque_mode()
        from encos_hand.protocol import MIT_TORQUE_HALF_LSB_NM

        model = self._glove_model()
        return MappingProxyType({
            model.joint_motor_names[config.joint]: float(np.nextafter(
                limits[mid] - MIT_TORQUE_HALF_LSB_NM, 0.0,
            )) / abs(config.sign * config.scale)
            for mid, config in self.motors.items()
        })

    @property
    def model_force_torque_command_limits_nm(self) -> Mapping[str, float]:
        """含零附近编码档的SDK名义参数确认上限，不是连续目标的编码裕量。"""
        limits = self._require_force_torque_mode()
        model = self._glove_model()
        return MappingProxyType({
            model.joint_motor_names[config.joint]:
            limits[mid] / abs(config.sign * config.scale)
            for mid, config in self.motors.items()
        })

    def connect(self) -> Mapping[str, object]:
        self._require_not_closed()
        if self._connected:
            raise RuntimeError("Encos手套已经连接")
        hand = None
        try:
            hand = self._hand_factory()
            hand.connect()
            found = hand.scan()
        except BaseException as error:
            if hand is not None:
                try:
                    hand.close()
                except BaseException:
                    pass
            raise EncosHardwareUnavailable("Encos手套连接或扫描失败") from error
        self._hand = hand
        self._connected = True
        self._online_ids = _flatten_online_ids(found)
        return {
            "model": "encos_dexhand",
            "online_ids": self._online_ids,
            "mapping_calibrated": self.mapping_calibrated,
            "mapping_source": self.mapping_source,
        }

    def scan(self) -> tuple[int, ...]:
        self._require_connected()
        try:
            self._online_ids = _flatten_online_ids(self._hand.scan())
        except BaseException as error:
            raise EncosHardwareUnavailable("Encos手套扫描失败") from error
        return self._online_ids

    def home(self, *, timeout_s: float = 30.0) -> Mapping[int, float]:
        """显式首次取得输出权：servo回标定零位，新鲜实测均在0.15rad内才返回。"""
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("回零超时必须为有限正数")
        self._require_connected()
        self._require_position_mode()
        self._require_send_calibration()
        self._require_all_online()
        targets = {mid: config.zero_deg for mid, config in self.motors.items()}
        if any(not config.min_deg <= targets[mid] <= config.max_deg
               for mid, config in self.motors.items()):
            raise ValueError("标定零位必须位于原始轴软限位内")
        angles, _, _ = self._read_raw_state()
        finish_ns = time.monotonic_ns() + int(timeout_s * 1e9)

        def servo(values, speed):
            # 工作线程独立检查250ms租约；上层中断不能无限重发回零目标。
            expires = min(finish_ns, time.monotonic_ns() + 250_000_000)
            self._owned_output = True
            for mid, value in values.items():
                result = self._hand.set_position(
                    motor_id=mid, target_deg=value, speed_rpm=speed,
                    current_a=self.home_current_a[mid], deadline_ns=expires,
                )
                if result is False:
                    raise EncosHardwareUnavailable(f"motor {mid} 伺服回零发送失败")

        self._homed = False
        self._output_enabled = False
        self._target_seed = None
        self._position_cache = None
        try:
            # 保持阶段只能保持原位，不能把越界反馈截断后以0rpm先拉到边界。
            # 越界轴继续0A读取，随后直接以规定5rpm向零位运动。
            hold = {mid: value for mid, value in angles.items()
                    if self.motors[mid].min_deg <= value <= self.motors[mid].max_deg}
            servo(hold, 0.0)
            time.sleep(0.2)
            started_ns = time.monotonic_ns()
            while True:
                if time.monotonic_ns() >= finish_ns:
                    raise TimeoutError("MotorHand伺服回零超时")
                servo(targets, 5.0)
                angles, _, received_ns = self._read_raw_state()
                errors = {mid: abs(float(np.deg2rad(angles[mid] - target)))
                          for mid, target in targets.items()}
                if received_ns >= started_ns and max(errors.values()) <= 0.15:
                    self._homed = True
                    self._output_enabled = True
                    return errors
                time.sleep(0.05)
        except BaseException:
            self.disable()
            raise

    def enable(self) -> None:
        """初次使能先回零；同会话主导切换恢复时不重复回零、不先发目标。"""
        self._require_connected()
        self._require_position_mode()
        if not self._homed:
            self.home()
        else:
            self._require_send_calibration()
            self._require_all_online()
            self._output_enabled = True
            self._target_seed = self._feedback_seed
            self._position_cache = None

    def enable_force_feedback(self) -> None:
        """取得纯前馈力控输出权；按显式名义Nm模式或原真实Kt电流模式预检。"""
        self._require_connected()
        try:
            self._require_force_calibration()
            self._require_all_online()
            if self._force_torque_limits_by_id is None:
                self._current_torques((0.0,) * 12)
            else:
                self._nominal_torques((0.0,) * 12)
            self._read_raw_state()
            if self._force_enabled:
                return
            if not self._homed:
                self.home()
            # 首次CAD预热不消耗随后MIT帧的250ms租约；正式帧仍重新读取真实反馈。
            self.prepare()
            self._output_enabled = True
            deadline_ns = time.monotonic_ns() + 250_000_000
            self._prepare_send(deadline_ns)
            angles, _, _ = self._read_raw_state()
            targets = self._force_targets(angles, deadline_ns)
            torques = (
                dict.fromkeys(self.motor_ids, 0.0)
                if self._force_torque_limits_by_id is None
                else self._project_nominal_torques(self._nominal_torques((0.0,) * 12), angles)
            )
            self._send_frame(targets, torques, deadline_ns, force_mode=True)
            self._force_enabled = True
            self._target_seed = None
            self._position_cache = None
        except BaseException:
            if self._owned_output:
                try:
                    self.disable()
                except BaseException:
                    pass
            raise

    def resume_force_feedback(self) -> None:
        """Resume a previously homed session without initiating another home."""
        self._require_connected()
        if not self._homed:
            raise RuntimeError("无法恢复力反馈：当前会话没有完成回零")
        self.enable_force_feedback()

    def prepare(self) -> None:
        """只读预热首次实际装配FK；该帧只作数值种子，控制前必须重新采样。"""
        self.read_state()

    def read_pose(self) -> GlovePose:
        return self.read_state().pose

    def read_state(self) -> EncosGloveState:
        self._require_connected()
        model = self._glove_model()
        angles, feedback, received_ns = self._read_raw_state()
        motors = self._model_motor_positions(angles, model)
        pose, solution = model.evaluate_motor_state(
            motors, timestamp_ns=received_ns, seed=self._feedback_seed
        )
        self._feedback_seed = solution
        self._position_cache = None
        return EncosGloveState(
            pose=pose,
            feedback_by_motor=feedback,
            online_ids=self._online_ids,
            received_ns=received_ns,
            raw_motor_positions_deg=angles,
            motor_positions_rad=solution.motor_positions,
            tree_positions_rad=solution.all_positions,
        )

    def constrain_position(self, pose: GlovePose) -> GlovePose:
        """返回真实可发姿态：q→机构IK→原始轴限位→机构FK，不返回未实现的请求q。"""
        self._require_connected()
        self._require_send_calibration()
        model = self._glove_model()
        self._position_targets(pose, None)
        _, targets, solution = self._position_cache
        return self._cache_position(targets, solution, timestamp_ns=pose.timestamp_ns)

    def position_tree(self, pose: GlovePose) -> Mapping[str, float]:
        """与限制后目标相匹配的同次完整CAD树，不重复机构正逆解。"""
        self._require_connected()
        self._require_send_calibration()
        self._position_targets(pose, None)
        return MappingProxyType(self._position_cache[2].all_positions)

    def position_from_tree(
        self, positions: Mapping[str, float], *, timestamp_ns: int | None = None,
    ) -> GlovePose:
        """预览同次逆解的完整目标树，不把已解出的q再逆解一次，不改变轨迹种子。"""
        self._require_connected()
        self._require_send_calibration()
        model = self._glove_model()
        solution = model.kinematics.evaluate_tree(positions)
        raw = self._raw_position_targets(solution.motor_positions, model)
        targets, solution = self._saturated_targets(raw, model, None, known_pose=solution)
        return self._cache_position(targets, solution, timestamp_ns=timestamp_ns)

    def position_from_motors(
        self, positions: Mapping[str, float], *, timestamp_ns: int | None = None,
    ) -> GlovePose:
        """模型电机轨迹→raw限位→真实机构FK，外部位置返回仍是q12。"""
        self._require_connected()
        self._require_send_calibration()
        model = self._glove_model()
        raw = self._raw_position_targets(positions, model)
        targets, solution = self._saturated_targets(raw, model, None)
        self._target_seed = solution
        return self._cache_position(targets, solution, timestamp_ns=timestamp_ns)

    def _cache_position(
        self, targets: dict[int, float], solution: Ve33Pose, *, timestamp_ns: int | None,
    ) -> GlovePose:
        actual = self._glove_model().to_glove_pose(solution, timestamp_ns=timestamp_ns)
        self._position_cache = (
            (tuple(actual.vector()), tuple(self.motors.values())), targets, solution
        )
        return actual

    def _raw_position_targets(
        self, positions: Mapping[str, float], model: Ve33GloveModel,
    ) -> dict[int, float]:
        if not isinstance(positions, Mapping) or set(positions) != set(model.motor_names):
            raise ValueError("模型电机目标必须按完整名称覆盖12轴")
        values = tuple(positions.values())
        if (any(np.iscomplexobj(value) or np.ndim(value) != 0 for value in values)
                or not np.isfinite(np.asarray(values, dtype=np.float64)).all()):
            raise ValueError("模型电机目标必须为有限实数rad")
        return {
            config.motor_id: float(config.zero_deg + np.rad2deg(
                positions[model.joint_motor_names[config.joint]]
            ) / config.sign / config.scale)
            for config in self.motors.values()
        }

    def validate_command(
        self, *, target_pose: GlovePose | None = None, current_a: tuple[float, ...] | None = None
    ) -> None:
        """只读完整预检，供跨设备控制循环在任何设备发令前调用；不生成期限。"""
        self._require_position_mode()
        if target_pose is None and current_a is None:
            raise ValueError("必须提供位置或触觉电流目标")
        self._require_connected()
        self._require_send_calibration()
        self._require_all_online()
        if current_a is not None:
            self._current_torques(current_a)
        angles, _, _ = self._read_raw_state()
        if target_pose is not None:
            self._position_targets(target_pose, None)
        else:
            self._saturated_targets(angles, self._glove_model(), None)

    def send_position(self, pose: GlovePose, *, deadline_ns: int) -> None:
        self._require_position_mode()
        self._prepare_send(deadline_ns)
        self._read_raw_state()
        self._validate_deadline(deadline_ns)
        targets = self._position_targets(pose, deadline_ns)
        self._send_frame(targets, dict.fromkeys(self.motor_ids, 0.0), deadline_ns)

    def send_haptic_current(
        self, current_a: tuple[float, ...], *, deadline_ns: int, target_pose: GlovePose | None = None
    ) -> None:
        """模型电机等效A→SDK原始轴Nm；上游已处理传动雅可比，不再投影。"""
        self._require_position_mode()
        self._prepare_send(deadline_ns)
        torques = self._current_torques(current_a)
        self._validate_deadline(deadline_ns)
        model = self._glove_model()
        angles, _, _ = self._read_raw_state()
        self._validate_deadline(deadline_ns)
        if target_pose is None:
            # 直接保持采样raw位置；不允许FK→IK的数值漂移。
            targets, _ = self._saturated_targets(angles, model, deadline_ns)
        else:
            targets = self._position_targets(target_pose, deadline_ns)
        self._send_frame(targets, torques, deadline_ns)

    def validate_force_current(self, current_a: tuple[float, ...]) -> None:
        """只读整帧预检；不取输出权，不修改传感数据或模式。"""
        self._require_force_current_mode()
        self._require_connected()
        self._require_force_calibration()
        self._require_all_online()
        self._current_torques(current_a)
        angles, _, _ = self._read_raw_state()
        self._force_targets(angles, None)

    def send_force_current(
        self, current_a: tuple[float, ...], *, deadline_ns: int,
    ) -> tuple[float, ...]:
        """纯MIT前馈；按实际模型电机边界投影，返回实际发送的模型电流。"""
        if not self._force_enabled:
            raise RuntimeError("必须先enable_force_feedback()取得力控输出权")
        try:
            self._prepare_send(deadline_ns)
            self._require_force_current_mode()
            self._require_force_calibration()
            # 先验证完整请求，不允许通过边界投影掩盖超限或非法电流。
            self._current_torques(current_a)
            angles, _, _ = self._read_raw_state()
            targets = self._force_targets(angles, deadline_ns)
            model = self._glove_model()
            values = dict(zip(model.motor_names, current_a))
            for mid, config in self.motors.items():
                name = model.joint_motor_names[config.joint]
                raw_current = values[name] * config.sign * config.scale
                if ((angles[mid] <= config.min_deg and raw_current < 0.0)
                        or (angles[mid] >= config.max_deg and raw_current > 0.0)):
                    values[name] = 0.0
            actual = tuple(float(values[name]) for name in model.motor_names)
            self._send_frame(
                targets, self._current_torques(actual), deadline_ns, force_mode=True,
            )
            return actual
        except BaseException:
            if self._owned_output:
                try:
                    self.disable()
                except BaseException:
                    pass
            raise

    def validate_force_torque(self, model_torque_nm: tuple[float, ...]) -> None:
        """只读完整帧预检；名义Nm不是实测力矩，也不声明实际电流上限。"""
        self._require_force_torque_mode()
        self._require_connected()
        self._require_force_calibration()
        self._require_all_online()
        self._nominal_torques(model_torque_nm)
        angles, _, _ = self._read_raw_state()
        self._force_targets(angles, None)

    def send_force_torque(
        self, model_torque_nm: tuple[float, ...], *, deadline_ns: int,
    ) -> tuple[float, ...]:
        """发送kp=kd=0名义Nm，返回SDK量化前参数按模型轴换算的确认值。

        raw边界向外目标投影至零附近向内编码档；其他连续目标不吸附编码网格，
        避免上层渐增／斜率历史被量化卡住。确认值不是解码值或实测力矩。
        """
        if not self._force_enabled:
            raise RuntimeError("必须先enable_force_feedback()取得力控输出权")
        try:
            self._require_force_torque_mode()
            self._prepare_send(deadline_ns)
            self._require_force_calibration()
            torques = self._nominal_torques(model_torque_nm)
            angles, _, _ = self._read_raw_state()
            targets = self._force_targets(angles, deadline_ns)
            torques = self._project_nominal_torques(torques, angles)
            model = self._glove_model()
            actual = {
                model.joint_motor_names[config.joint]:
                torques[mid] / (config.sign * config.scale)
                for mid, config in self.motors.items()
            }
            self._send_frame(targets, torques, deadline_ns, force_mode=True)
            return tuple(float(actual[name]) for name in model.motor_names)
        except BaseException:
            if self._owned_output:
                try:
                    self.disable()
                except BaseException:
                    pass
            raise

    def _nominal_torques(self, model_torque_nm: tuple[float, ...]) -> dict[int, float]:
        self._require_force_torque_mode()
        if any(
            isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or np.iscomplexobj(value)
            for value in model_torque_nm
        ):
            raise ValueError("model_torque_nm必须是有限实数的12维名义Nm数组")
        values = np.asarray(model_torque_nm, dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("model_torque_nm必须是有限实数的12维名义Nm数组")
        model = self._glove_model()
        limits = self.model_force_torque_limits_nm
        named = dict(zip(model.motor_names, values))
        for name in model.motor_names:
            if abs(named[name]) > limits[name]:
                raise ValueError(f"{name} 名义力矩超出Encos软件限幅（含编码裕量）")
        return {
            mid: float(named[model.joint_motor_names[config.joint]] * config.sign * config.scale)
            for mid, config in self.motors.items()
        }

    def _project_nominal_torques(
        self, torques: Mapping[int, float], angles: Mapping[int, float],
    ) -> dict[int, float]:
        from encos_hand.protocol import MIT_TORQUE_HALF_LSB_NM

        projected = dict(torques)
        for mid, config in self.motors.items():
            # 12bit对称量程没有精确零档；边界处包括微小向内请求也不能round成向外。
            if angles[mid] <= config.min_deg:
                projected[mid] = max(projected[mid], MIT_TORQUE_HALF_LSB_NM)
            elif angles[mid] >= config.max_deg:
                projected[mid] = min(projected[mid], -MIT_TORQUE_HALF_LSB_NM)
        return projected

    def _require_force_torque_mode(self) -> Mapping[int, float]:
        if self._force_torque_limits_by_id is None:
            raise RuntimeError("名义Nm模式需要显式force_torque_limits_nm；不能用协议量程替代")
        return self._force_torque_limits_by_id

    def _require_force_current_mode(self) -> None:
        if self._force_torque_limits_by_id is not None:
            raise RuntimeError("显式名义Nm模式禁止force current接口；请使用force torque接口")

    def _force_targets(
        self, angles: Mapping[int, float], deadline_ns: int | None,
    ) -> dict[int, float]:
        # 验证实测完整CAD机构而非裁剪后的虚构姿态；kp=kd=0时位置仅满足SDK格式。
        self._check_preparation_deadline(deadline_ns)
        model = self._glove_model()
        motors = self._model_motor_positions(angles, model)
        seed = self._nearest_seed(motors, motor_space=True)
        if seed is None or any(seed.motor_positions[name] != value for name, value in motors.items()):
            model.kinematics.evaluate_motors(motors, seed=seed)
        self._check_preparation_deadline(deadline_ns)
        return {
            mid: float(np.clip(angles[mid], config.min_deg, config.max_deg))
            for mid, config in self.motors.items()
        }

    def _current_torques(self, current_a: tuple[float, ...]) -> dict[int, float]:
        if self._torque_constants_by_id is None:
            raise EncosCalibrationRequired("缺少Encos电机 torque_constant_nm_per_a 标定")
        values = np.asarray(current_a, dtype=np.float64)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise ValueError("current_a必须是有限的12维数组")
        model = self._glove_model()
        indices = {name: index for index, name in enumerate(model.motor_names)}
        torques = {}
        for config in self.motors.values():
            model_name = model.joint_motor_names[config.joint]
            raw_current = float(values[indices[model_name]] * config.sign * config.scale)
            if not np.isfinite(raw_current) or abs(raw_current) > min(
                self.max_haptic_current_a, config.current_limit_a
            ):
                raise ValueError(f"motor {config.motor_id} 原始轴触觉电流超出Encos安全限幅")
            torque = raw_current * self._torque_constants_by_id[config.motor_id]
            if not np.isfinite(torque):
                raise ValueError(f"motor {config.motor_id} 前馈力矩必须有限")
            torques[config.motor_id] = torque
        return torques

    def _check_preparation_deadline(self, deadline_ns: int | None) -> None:
        if deadline_ns is not None:
            self._validate_deadline(deadline_ns)

    def _glove_model(self) -> Ve33GloveModel:
        if self._model is None:
            self._model = Ve33GloveModel(self.hand, asset_root=self.asset_root)
        return self._model

    def _prepare_send(self, deadline_ns: int) -> None:
        self._require_connected()
        self._validate_deadline(deadline_ns)
        self._require_send_calibration()
        self._require_all_online()
        if not self._homed or not self._output_enabled:
            raise RuntimeError("MotorHand首次输出必须先home()；失能后先enable()恢复本会话输出")

    def _read_raw_state(self) -> tuple[dict[int, float], dict[int, dict[str, object]], int]:
        self._require_connected()
        self._require_all_online()
        try:
            feedback = self._hand.get_all_feedbacks()
            checked_ns = time.monotonic_ns()
        except Exception as error:
            raise EncosHardwareUnavailable("读取Encos手套状态失败") from error
        if not isinstance(feedback, Mapping):
            raise EncosHardwareUnavailable("Encos反馈必须是电机ID映射")
        normalized = {int(mid): dict(value) for mid, value in feedback.items() if isinstance(value, Mapping)}
        missing = set(self.motor_ids) - set(normalized)
        if missing:
            raise EncosHardwareUnavailable(f"Encos反馈缺少电机: {sorted(missing)}")
        raw = {}
        stamps = []
        for mid in self.motor_ids:
            diagnostic = normalized[mid]
            stamp = diagnostic.get("received_ns")
            if isinstance(stamp, bool) or not isinstance(stamp, int):
                raise EncosHardwareUnavailable(f"motor {mid} 缺少真实接收时戳，禁止使用缓存读出时间替代")
            if not 0 <= checked_ns - stamp <= self.feedback_timeout_ns:
                raise EncosHardwareUnavailable(
                    f"motor {mid} 反馈已过期或时钟域错误"
                    f"（反馈年龄={(checked_ns - stamp) / 1e6:.1f}ms，"
                    f"上限={self.feedback_timeout_ns / 1e6:.1f}ms）"
                )
            stamps.append(stamp)
            if "angle_deg" not in diagnostic:
                raise EncosHardwareUnavailable(f"motor {mid} 缺少原始角度反馈")
            raw[mid] = float(diagnostic["angle_deg"])
            if not np.isfinite(raw[mid]):
                raise ValueError(f"motor {mid} 反馈角度必须有限")
            diagnostic = normalized[mid]
            if diagnostic.get("err_code", 0) != 0:
                raise EncosHardwareUnavailable(f"motor {mid} 报警: {diagnostic}")
            for key in ("angle_deg", "current_a", "speed_rpm", "temperature"):
                if key in diagnostic and not np.isfinite(float(diagnostic[key])):
                    raise ValueError(f"motor {mid} {key}反馈必须有限")
        return raw, normalized, min(stamps)

    def _model_motor_positions(self, raw: Mapping[int, float], model: Ve33GloveModel) -> dict[str, float]:
        return {
            model.joint_motor_names[config.joint]:
            float(np.deg2rad((raw[config.motor_id] - config.zero_deg) * config.sign * config.scale))
            for config in self.motors.values()
        }

    def _nearest_seed(
        self, positions: Mapping[str, float], *, motor_space: bool,
        known_pose: Ve33Pose | None = None,
    ) -> Ve33Pose | None:
        """只选数值初值；微小限幅优先复用本次已验证树，不从远处反馈重新延拓。"""
        candidates = [
            seed for seed in (known_pose, self._feedback_seed, self._target_seed)
            if seed is not None
        ]
        if not candidates:
            return None
        kinematics = self._glove_model().kinematics
        names = kinematics.motor_names if motor_space else kinematics.joint_names
        coupled = [name for name, selected in zip(names, kinematics._continuation_inputs) if selected]
        def distance(seed):
            values = seed.motor_positions if motor_space else seed.joint_positions
            # 无被动耦合的直驱大角差不应压过真正昂贵的闭链距离。
            return max((abs(positions[name] - values[name]) for name in coupled), default=0.0)
        return min(candidates, key=distance)

    def _position_targets(self, pose: GlovePose, deadline_ns: int | None) -> dict[int, float]:
        if not isinstance(pose, GlovePose):
            raise TypeError("pose必须是GlovePose")
        model = self._glove_model()
        self._check_preparation_deadline(deadline_ns)
        key = (tuple(pose.vector()), tuple(self.motors.values()))
        if self._position_cache is not None and self._position_cache[0] == key:
            return self._position_cache[1]
        internal = {f"{name}_joint": value for name, value in pose.joints.items()}
        seed = self._nearest_seed(internal, motor_space=False)
        if seed is None:
            solution = model.evaluate_raw(dict(pose.joints))
        else:
            nearby = all(
                abs(internal[name] - seed.joint_positions[name]) <= 0.04 + 1e-12
                for name, selected in zip(model.kinematics.joint_names, model.kinematics._continuation_inputs)
                if selected
            )
            solution = model.kinematics.evaluate(internal, seed=seed, local=nearby)
        self._check_preparation_deadline(deadline_ns)
        raw = self._raw_position_targets(solution.motor_positions, model)
        targets, solution = self._saturated_targets(raw, model, deadline_ns, known_pose=solution)
        self._target_seed = solution
        self._position_cache = (key, targets, solution)
        return targets

    def _saturated_targets(
        self, raw: Mapping[int, float], model: Ve33GloveModel, deadline_ns: int | None,
        *, known_pose: Ve33Pose | None = None,
    ) -> tuple[dict[int, float], Ve33Pose]:
        if set(raw) != set(self.motor_ids) or not np.isfinite(list(raw.values())).all():
            raise ValueError("目标必须完整覆盖12个有限raw电机角度")
        targets = {
            config.motor_id: float(np.clip(raw[config.motor_id], config.min_deg, config.max_deg))
            for config in self.motors.values()
        }
        # 正常触边保留控制；饱和后的实际机构仍须可解，否则整帧拒绝。
        if known_pose is not None and all(targets[mid] == raw[mid] for mid in self.motor_ids):
            # 未发生饱和，q→m求解已经同时证明此完整CAD树，不再解一次相同机构。
            solution = known_pose
        else:
            motors = self._model_motor_positions(targets, model)
            solution = model.kinematics.evaluate_motors(
                motors, seed=self._nearest_seed(
                    motors, motor_space=True, known_pose=known_pose,
                )
            )
        self._check_preparation_deadline(deadline_ns)
        return targets, solution

    def _send_frame(
        self, targets: Mapping[int, float], torques: Mapping[int, float], deadline_ns: int,
        *, force_mode: bool = False,
    ) -> None:
        if set(targets) != set(self.motor_ids) or set(torques) != set(self.motor_ids):
            raise ValueError("MIT帧必须完整覆盖12轴")
        for config in self.motors.values():
            target, torque = targets[config.motor_id], torques[config.motor_id]
            if not np.isfinite((target, torque)).all() or not config.min_deg <= target <= config.max_deg:
                raise ValueError(f"motor {config.motor_id} MIT帧非法")
        self._require_all_online()
        self._validate_deadline(deadline_ns)
        sent = False
        try:
            for mid in self.motor_ids:
                self._validate_deadline(deadline_ns)
                sent = True
                self._owned_output = True
                result = self._hand.set_mit(
                    motor_id=mid, target_deg=targets[mid],
                    kp=0.0 if force_mode else self.haptic_kp,
                    kd=0.0 if force_mode else self.haptic_kd,
                    vel_rad_s=0.0, torque_nm=torques[mid], deadline_ns=deadline_ns,
                )
                if result is False:
                    raise EncosHardwareUnavailable(f"motor {mid} MIT发送失败")
        except BaseException:
            if sent:
                try:
                    self.disable()
                except BaseException:
                    pass
            raise


    def disable(self) -> None:
        self._require_connected()
        self._output_enabled = False
        self._force_enabled = False
        self._target_seed = None
        self._position_cache = None
        if not self._owned_output:
            return
        first_error = None
        for motor_id in self._online_ids:
            try:
                self._hand.enable(motor_id, False)
            except BaseException as error:
                first_error = first_error or error
        time.sleep(0.05)
        self._owned_output = first_error is not None
        if first_error is not None:
            raise EncosHardwareUnavailable("Encos手套失能失败") from first_error

    def emergency_stop(self) -> None:
        self.disable()

    def close(self) -> None:
        if self._closed:
            return
        hand = self._hand
        try:
            if hand is not None and self._owned_output:
                self.disable()
        finally:
            self._closed = True
            self._force_enabled = False
            self._output_enabled = False
            self._hand = None
            self._connected = False
            self._online_ids = ()
            if hand is not None:
                hand.close()

    def _require_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Encos手套适配器已关闭")

    def _require_connected(self) -> None:
        self._require_not_closed()
        if not self._connected or self._hand is None:
            raise RuntimeError("Encos手套尚未连接")

    def _require_position_mode(self) -> None:
        if self._force_enabled:
            raise RuntimeError("力控模式禁止位置输出；先disable()再enable()切换位置模式")

    def _require_send_calibration(self) -> None:
        if self._require_calibration_on_send:
            raise EncosCalibrationRequired(
                f"Encos电机映射尚未标定: {self.mapping_source}; 先完成轴序、零位、符号和限位标定"
            )

    def _require_force_calibration(self) -> None:
        # 只读调试的allow_uncalibrated不能放宽真实力控输出资格。
        if not self.mapping_calibrated:
            raise EncosCalibrationRequired("力反馈需要真实标定的Encos轴序、零位、方向和限位")
        self._require_send_calibration()

    def _require_all_online(self) -> None:
        if self._hand is not None and hasattr(self._hand, "get_online_ids"):
            try:
                self._online_ids = tuple(sorted(int(mid) for mid in self._hand.get_online_ids()))
            except Exception as error:
                raise EncosHardwareUnavailable("读取Encos在线状态失败") from error
        missing = sorted(set(self.motor_ids) - set(self._online_ids))
        if missing:
            raise EncosHardwareUnavailable(f"Encos手套缺少在线电机: {missing}")

    @staticmethod
    def _validate_deadline(deadline_ns: int) -> None:
        if isinstance(deadline_ns, bool) or not isinstance(deadline_ns, int):
            raise ValueError("deadline_ns必须是整数")
        if not 0 <= deadline_ns <= 2**64 - 1:
            raise ValueError("deadline_ns必须在uint64范围内")
        if deadline_ns <= time.monotonic_ns():
            raise TimeoutError("Encos手套目标期限已过期")


def _default_mapping_path() -> Path:
    return Path(__file__).resolve().parents[4] / "hardware" / "encos_dexhand" / "configs" / "ffb" / "glove_mapping.json"


def _default_hand_factory():
    try:
        from encos_hand import EncosHand
    except ImportError as error:
        raise EncosHardwareUnavailable("无法导入已部署的 encos_hand SDK") from error
    return EncosHand()


def _mapping_config(
    mapping: Mapping[int, Mapping[str, object]] | None,
    path: Path,
) -> Mapping[str, object]:
    if mapping is not None:
        return {"calibrated": True, "mapping_source": "injected", "motors": {str(k): v for k, v in mapping.items()}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"Encos映射配置不存在: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Encos映射配置不是合法JSON: {path}") from error
    if not isinstance(data, Mapping):
        raise ValueError("Encos映射配置根节点必须是对象")
    return data


def _parse_motor_configs(raw: object) -> dict[int, EncosMotorConfig]:
    if not isinstance(raw, Mapping):
        raise ValueError("Encos motors配置必须是对象")
    configs: dict[int, EncosMotorConfig] = {}
    for raw_id, value in raw.items():
        motor_id = int(raw_id)
        if not isinstance(value, Mapping):
            raise ValueError(f"motor {motor_id} 配置必须是对象")
        configs[motor_id] = EncosMotorConfig(
            motor_id=motor_id,
            joint=str(value["joint"]),
            zero_deg=float(value.get("zero_deg", 0.0)),
            sign=float(value.get("sign", 1.0)),
            scale=float(value.get("scale", 1.0)),
            min_deg=float(value.get("min_deg", -360.0)),
            max_deg=float(value.get("max_deg", 360.0)),
            speed_rpm=float(value.get("speed_rpm", 30.0)),
            current_limit_a=float(value.get("current_limit_a", 0.8)),
        )
    if set(configs) != set(range(1, 13)):
        raise ValueError("Encos motors必须完整覆盖1到12号电机")
    if {config.joint for config in configs.values()} != set(GLOVE_JOINTS):
        raise ValueError("Encos motors必须完整覆盖12个手套独立坐标")
    return dict(sorted(configs.items()))


def _flatten_online_ids(found: object) -> tuple[int, ...]:
    if not isinstance(found, Mapping):
        raise ValueError("Encos scan结果必须是通道到电机ID列表的映射")
    ids = {int(motor_id) for values in found.values() for motor_id in values}
    return tuple(sorted(ids))




def _normalize_torque_constants(value: object) -> Mapping[int, float] | None:
    """仅接受真实的正标量或完整raw ID1..12标定，归一化后不可变。"""
    if value is None:
        return None
    if isinstance(value, Mapping):
        constants = {}
        for key, constant in value.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise ValueError("Kt映射键必须为raw电机ID1..12")
            try:
                motor_id = int(key)
            except ValueError as error:
                raise ValueError("Kt映射键必须为raw电机ID1..12") from error
            if motor_id in constants:
                raise ValueError("Kt映射不能包含重复raw电机ID")
            constants[motor_id] = constant
        if set(constants) != set(range(1, 13)):
            raise ValueError("Kt映射必须完整覆盖raw电机ID1..12")
    else:
        constants = dict.fromkeys(range(1, 13), value)
    normalized = {}
    for motor_id, constant in constants.items():
        if isinstance(constant, bool) or np.ndim(constant) != 0 or np.iscomplexobj(constant):
            raise ValueError("torque_constant_nm_per_a必须为正有限值")
        number = float(constant)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError("torque_constant_nm_per_a必须为正有限值")
        normalized[motor_id] = number
    return MappingProxyType(normalized)


def _normalize_force_torque_limits(value: object) -> Mapping[int, float] | None:
    """显式raw轴名义Nm软件cap；协议编码边界绝非硬件最大力矩。"""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("force_torque_limits_nm必须完整映射raw电机ID1..12")
    limits = {}
    for key, limit in value.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError("force_torque_limits_nm键必须为raw电机ID1..12")
        try:
            motor_id = int(key)
        except ValueError as error:
            raise ValueError("force_torque_limits_nm键必须为raw电机ID1..12") from error
        if motor_id in limits:
            raise ValueError("force_torque_limits_nm不能包含重复raw电机ID")
        if isinstance(limit, (bool, np.bool_)) or np.ndim(limit) != 0 or np.iscomplexobj(limit):
            raise ValueError("force_torque_limits_nm必须为正有限名义Nm")
        limits[motor_id] = float(limit)
    if set(limits) != set(range(1, 13)):
        raise ValueError("force_torque_limits_nm必须完整覆盖raw电机ID1..12")
    from encos_hand.protocol import MIT_TORQUE_HALF_LSB_NM, MIT_TORQUE_MAX_NM

    for limit in limits.values():
        if not np.isfinite(limit) or not MIT_TORQUE_HALF_LSB_NM < limit <= MIT_TORQUE_MAX_NM:
            raise ValueError("force_torque_limits_nm必须大于编码半LSB且不超过协议编码量程")
    return MappingProxyType(limits)
