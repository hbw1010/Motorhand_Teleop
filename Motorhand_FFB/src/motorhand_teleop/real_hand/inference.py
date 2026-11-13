"""Native inference session; Python supplies model targets and schedules cycles."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import math
import os
from pathlib import Path

from ..bidirectional import ControlMode

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LIBRARY = _PROJECT_ROOT / ".cache/native-control/build/libinference_bridge.so"


class _Options(ct.Structure):
    _fields_ = [(name, ct.c_char_p) for name in (
        "asset_root", "retarget_yaml", "glove_mapping", "linear_feedback",
        "contact_feedback", "wuji_serial", "approved_firmware") ] + [
        ("left", ct.c_int), ("transition_ns", ct.c_uint64)]


class _Identity(ct.Structure):
    _fields_ = [(name, ct.c_char * 256) for name in ("serial", "address", "firmware")] + [("left", ct.c_int)]


class _State(ct.Structure):
    _fields_ = [(name, ct.c_double * 20) for name in ("positions_rad", "velocities_rad_s", "currents_a")] + [("received_ns", ct.c_uint64)]


class _Tick(ct.Structure):
    _fields_ = [("hand", _State), ("wuji_target_rad", ct.c_double * 20),
                ("glove_output", ct.c_double * 12), ("virtual_hand_nm", ct.c_double * 20)] + [
        (name, ct.c_uint64) for name in ("target_deadline_ns", "glove_output_source_ns", "cycle_started_ns",
                                          "wuji_submit_ns", "glove_submit_ns", "cycle_completed_ns", "jacobian_source_ns") ] + [
        (name, ct.c_int) for name in ("takeover", "inverse_fresh", "force_geometry_valid", "force_commanded", "contact_phase") ] + [("contact", ct.c_ubyte * 20)]


@dataclass(frozen=True)
class WujiState:
    """Measured joint position, velocity, current, and original monotonic receive time."""
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    effort_a: tuple[float, ...]
    received_ns: int
    online_count: int = 20


def _state(raw: _State) -> WujiState:
    return WujiState(tuple(raw.positions_rad), tuple(raw.velocities_rad_s), tuple(raw.currents_a), raw.received_ns)


@dataclass(frozen=True)
class InferenceTick:
    """Completed hand command and, if available, completed asynchronous glove output.

    Glove output is motor radians in inference or nominal motor Nm in takeover;
    virtual hand torque is software-estimated, not measured. None means unavailable.
    """
    hand2_state: WujiState
    sent_positions_rad: tuple[float, ...]
    glove_output: tuple[float, ...] | None
    virtual_hand_nm: tuple[float, ...] | None
    takeover: bool
    inverse_fresh: bool
    force_geometry_valid: bool
    force_commanded: bool
    contact_phase: int
    contact: tuple[bool, ...] | None
    target_deadline_ns: int
    glove_output_source_ns: int
    cycle_started_ns: int
    wuji_submit_ns: int
    glove_submit_ns: int
    cycle_completed_ns: int
    jacobian_source_ns: int


def _tick(t: _Tick) -> InferenceTick:
    return InferenceTick(_state(t.hand), tuple(t.wuji_target_rad), tuple(t.glove_output) if t.glove_submit_ns else None,
        tuple(t.virtual_hand_nm) if t.force_commanded and t.force_geometry_valid else None,
        bool(t.takeover), bool(t.inverse_fresh), bool(t.force_geometry_valid), bool(t.force_commanded),
        t.contact_phase, tuple(map(bool, t.contact)) if t.force_commanded and t.force_geometry_valid else None,
        t.target_deadline_ns, t.glove_output_source_ns, t.cycle_started_ns, t.wuji_submit_ns,
        t.glove_submit_ns, t.cycle_completed_ns, t.jacobian_source_ns)


def _load_library():
    lib = ct.CDLL(os.environ.get("MOTORHAND_INFERENCE_LIBRARY", str(_LIBRARY)))
    handle = ct.c_void_p
    signatures = {
        "create": [ct.POINTER(_Options), ct.POINTER(handle)],
        "connect": [handle, ct.POINTER(_Identity)], "read": [handle, ct.POINTER(_State)],
        "start": [handle], "request_takeover": [handle, ct.c_int],
        "step": [handle, ct.POINTER(ct.c_double), ct.c_size_t, ct.c_uint64, ct.POINTER(_Tick)],
        "step_takeover": [handle, ct.POINTER(_Tick)], "stop": [handle], "destroy": [handle],
    }
    for name, args in signatures.items():
        fn = getattr(lib, "mh_inference_" + name)
        fn.argtypes = args
        fn.restype = None if name == "destroy" else ct.c_int
    lib.mh_inference_error.argtypes = []
    lib.mh_inference_error.restype = ct.c_char_p
    return lib


class InferenceSession:
    """Single-threaded native session. Connect is read-only; start explicitly enables.

    The caller schedules every cycle and must provide the model's original monotonic
    production timestamp. No background Python control loop is started.
    """
    def __init__(self, options: _Options):
        self._lib = _load_library()
        self._handle = ct.c_void_p()
        self._closed = False
        self._started = False
        self._mode = ControlMode.INFERENCE
        self._check(self._lib.mh_inference_create(ct.byref(options), ct.byref(self._handle)))

    def _check(self, result):
        if result:
            self._started = False
            raise RuntimeError((self._lib.mh_inference_error() or b"native inference error").decode(errors="replace"))

    def _abort(self):
        if self._handle and not self._closed:
            result = self._lib.mh_inference_stop(self._handle)
            self._started = False
            self._closed = True
            if result:
                raise RuntimeError("异常后停止失败：" + (self._lib.mh_inference_error() or b"unknown error").decode(errors="replace"))

    @property
    def mode(self) -> ControlMode:
        """Requested control owner; transitions are performed by the native session."""
        return self._mode

    def connect(self):
        """Connect and inspect devices without homing or enabling them."""
        try:
            identity = _Identity()
            self._check(self._lib.mh_inference_connect(self._handle, ct.byref(identity)))
            return {"serial_number": identity.serial.decode(), "address": identity.address.decode(),
                    "firmware": identity.firmware.decode(), "hand": "left" if identity.left else "right"}
        except BaseException:
            self._abort()
            raise

    def read_wuji_state(self) -> WujiState:
        """Read all 20 measured positions, velocities and currents with their receive time."""
        try:
            state = _State()
            self._check(self._lib.mh_inference_read(self._handle, ct.byref(state)))
            return _state(state)
        except BaseException:
            self._abort()
            raise

    def start(self) -> None:
        """Explicitly home the glove and enable both devices."""
        try:
            self._check(self._lib.mh_inference_start(self._handle))
            self._started = True
        except BaseException:
            self._abort()
            raise

    def step(self, positions_rad=None, *, produced_at_ns: int | None = None) -> InferenceTick:
        """Execute one native cycle; inference requires 20 rad targets and their source time.

        In takeover call step() with no target. Source time is time.monotonic_ns()
        captured when the model produced the target, not the time of submission.
        """
        try:
            result = _Tick()
            if self._mode is ControlMode.GLOVE_TAKEOVER:
                if positions_rad is not None or produced_at_ns is not None:
                    raise ValueError("takeover does not accept inference targets")
                self._check(self._lib.mh_inference_step_takeover(self._handle, ct.byref(result)))
            else:
                if positions_rad is None or isinstance(produced_at_ns, bool) or not isinstance(produced_at_ns, int) or not 0 < produced_at_ns < 2**64:
                    raise ValueError("inference requires 20 targets and their original produced_at_ns")
                values = tuple(float(x) for x in positions_rad)
                if len(values) != 20 or not all(map(math.isfinite, values)):
                    raise ValueError("inference target must contain 20 finite positions")
                target = (ct.c_double * 20)(*values)
                self._check(self._lib.mh_inference_step(self._handle, target, 20, produced_at_ns, ct.byref(result)))
            return _tick(result)
        except BaseException:
            self._abort()
            raise

    def _request(self, takeover: bool):
        try:
            self._check(self._lib.mh_inference_request_takeover(self._handle, int(takeover)))
            self._mode = ControlMode.GLOVE_TAKEOVER if takeover else ControlMode.INFERENCE
        except BaseException:
            self._abort()
            raise

    def take_over(self) -> None:
        """Request glove control and force feedback; continue calling step()."""
        self._request(True)

    def resume_inference(self) -> None:
        """Request a smooth return to model control."""
        self._request(False)

    def close(self) -> None:
        """Stop outputs and release devices; safe to call repeatedly."""
        if self._handle:
            try:
                if not self._closed:
                    self._check(self._lib.mh_inference_stop(self._handle))
            finally:
                self._lib.mh_inference_destroy(self._handle)
                self._handle = ct.c_void_p()
                self._closed = True
                self._started = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_inference_session(*, serial_number: str | None = None, address: str | None = None,
        approved_firmware: str = "2.2.3", hand: str = "right", glove_hand: str = "right",
        transition_s: float = 0.5, mapping_path: str | Path | None = None,
        asset_root: str | Path | None = None, feedback_path: str | Path | None = None,
        contact_path: str | Path | None = None) -> InferenceSession:
    """Construct without connecting or enabling; MOTORHAND_INFERENCE_LIBRARY overrides the .cache library."""
    if address is not None:
        raise ValueError("native session does not support a manually specified address")
    if hand not in ("left", "right") or glove_hand != hand:
        raise ValueError("native session requires matching left/right device sides")
    if not math.isfinite(transition_s) or not 0 < transition_s < (2**64 / 1e9):
        raise ValueError("transition_s must be positive and representable in nanoseconds")
    duration = int(transition_s * 1e9)
    if not duration:
        raise ValueError("transition_s is shorter than one nanosecond")
    paths = (asset_root or _PROJECT_ROOT.parent / "assets/hand_motor_ve33",
             _PROJECT_ROOT / "config/retargeting/official" / (hand + ".yaml"),
             mapping_path or _PROJECT_ROOT.parent / "hardware/encos_dexhand/configs/ffb/glove_mapping.json",
             feedback_path or _PROJECT_ROOT.parent / "config/feedback/wuji-max-torque-linear-right.json",
             contact_path or _PROJECT_ROOT.parent / "config/feedback/wuji-contact-impedance-right.json")
    strings = [os.fsencode(p) for p in paths] + [(serial_number or "").encode(), approved_firmware.encode()]
    return InferenceSession(_Options(*strings, int(hand == "left"), duration))
