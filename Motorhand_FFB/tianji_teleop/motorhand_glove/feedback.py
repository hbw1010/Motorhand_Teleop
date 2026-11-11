"""Motorhand force feedback for one glove from a Wuji Hand 2 topic stream.

The algorithm is this project's ForceFeedbackController in contact-impedance
mode: measured current, final-target error and speed identify a blocked finger;
a virtual spring/damper projected through the retarget Jacobian produces
nominal torque, capped and slew-limited by the selected asset. The ROS2 node
forwards the hand state and final target; no Tianji workspace package is needed.

It reflects only while the hand follows this glove (the node: its final target is one this glove sent); each
time the glove takes the hand again it starts from zero, as Motorhand's runtime does when it hands the Wuji hand
to the glove.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from motorhand_teleop.domain import WujiState
from motorhand_teleop.real_hand.contact_feedback import ContactFeedbackConfig
from motorhand_teleop.real_hand.force_feedback import ForceFeedbackController
from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig
from motorhand_teleop.real_hand.runtime import RetargetPipelineFactory, create_retargeter


def load_limits(path, *, serial: str | None = None) -> MaxTorqueLinearConfig:
    """The limits config (the glove's nominal torque caps and slew, the Wuji hand it is for); serial: that hand's."""
    limits = MaxTorqueLinearConfig.from_json(Path(path))
    if serial:
        limits.validate_serial(serial)
    return limits


def default_contact(root, side: str) -> Path:
    """Motorhand's own contact-impedance parameters for that side."""
    return Path(root) / "config" / "feedback" / f"contact-impedance-{side}.json"


def wuji_state(message: dict, *, serial: str | None = None) -> WujiState:
    """The node's {"ns", "q", "dq", "effort"} (the driver's joint_states in the 20 command joints' order, received at
    monotonic ns) as Motorhand's WujiState; with serial, the controller checks it is the limits config's hand."""
    return WujiState(np.asarray(message["q"], dtype=np.float64), np.asarray(message["dq"], dtype=np.float64),
                     np.asarray(message["effort"], dtype=np.float64), int(message["ns"]), 20,
                     diagnostics={"serial_number": serial} if serial else {})


class Feedback:
    """One glove's force feedback. limits: a MaxTorqueLinearConfig or its JSON; contact: a ContactFeedbackConfig, its
    JSON, or None for Motorhand's own for this side."""

    def __init__(self, root, side: str, glove_driver, *, limits, contact=None, serial: str | None = None,
                 clock_ns=time.monotonic_ns) -> None:
        if not isinstance(limits, MaxTorqueLinearConfig):
            limits = load_limits(limits, serial=serial)
        if contact is None:
            contact = default_contact(root, side)
        if not isinstance(contact, ContactFeedbackConfig):
            contact = ContactFeedbackConfig.from_json(Path(contact))
        self.limits, self.contact = limits, contact
        self._controller_ = self._controller(root, side, glove_driver, limits, contact, clock_ns)
        self._prepared = self._owned = False

    @staticmethod
    def _controller(root, side, glove_driver, limits, contact, clock_ns):
        return ForceFeedbackController(
            limits, create_retargeter(hand=side, glove_hand=side, project_root=root), glove_driver,
            RetargetPipelineFactory(hand=side, glove_hand=side, project_root=Path(root)),
            clock_ns=clock_ns, contact_config=contact,
        )

    def prepare(self, hand: WujiState, glove, *, now_ns: int) -> None:
        """The first Jacobian, at this pose (waits for it)."""
        self._controller_.prepare(hand, glove, now_ns=now_ns)
        self._prepared = True

    def begin(self, *, now_ns: int) -> None:
        """The glove has the hand from now: no reaction yet, the slew history at zero."""
        self._controller_.reset(now_ns=now_ns)
        self._owned = True

    def step(self, hand: WujiState, glove, target, *, owned: bool, now_ns: int, deadline_ns: int):
        """Motorhand's ForceFeedbackCommand for this frame (model_torque_nm: the glove's 12), or None while the hand
        does not follow this glove."""
        if not owned:
            self._owned = False
            return None
        if not self._prepared:
            self.prepare(hand, glove, now_ns=now_ns)
        if not self._owned:
            self.begin(now_ns=now_ns)
        return self._controller_.compute(hand, glove, now_ns=now_ns, deadline_ns=deadline_ns,
                                         target_positions_rad=target)

    def acknowledge(self, torque_nm) -> None:
        """The torques that went out (or would have: a preview), for the slew limit's history."""
        self._controller_.acknowledge_torque(tuple(float(t) for t in torque_nm))

    def close(self) -> None:
        self._controller_.close()
