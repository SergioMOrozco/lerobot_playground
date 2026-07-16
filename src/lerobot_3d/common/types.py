"""Shared data types referenced across the point-cloud/teleop pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Datapoint:
    """One camera's frame: raw color/depth plus what's needed to fuse it into world frame."""

    serial: str
    color: np.ndarray | None
    depth: np.ndarray
    depth_scale: float
    max_depth: float
    X_WC: np.ndarray
    color_intrinsics: object  # pyrealsense2.intrinsics-like: needs .fx/.fy/.ppx/.ppy
    obj_mask: np.ndarray | None = None
    joint_positions: dict[str, float] | None = None
    """Raw motor-space observation (``"<joint>.pos"`` keys) from the follower that produced
    this step's robot state -- same format as ``Follower.get_observation()``/``send_action()``,
    so it can be fed straight into ``RobotState`` or a new action without reconversion."""
