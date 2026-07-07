"""Shared USB / device identifiers for SO101 teleop stacks (no I/O imports).

Actual device values (ports, ids, RealSense serials) live in ``hardware_config.yaml``,
not here -- see :func:`load_hardware_devices`. Edit that file when your USB layout changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from lerobot_playground.paths import resolve_hardware_config_yaml


@dataclass(frozen=True)
class SO101AxisConfig:
    """One SO101 arm on a serial device (leader teleop or follower robot)."""

    port: str
    """Device path, e.g. ``/dev/ttyACM0``."""
    id: str
    """LeRobot calibration / bus id, e.g. ``bender_leader_arm``."""


HARDWARE_CONFIG_YAML = "hardware_config.yaml"
"""Default filename for the editable device map; resolved via
:func:`lerobot_playground.paths.resolve_hardware_config_yaml`."""


@dataclass(frozen=True)
class HardwareDevices:
    """Leader/follower ports+ids and RealSense serials, as loaded from YAML."""

    leaders: tuple[SO101AxisConfig, ...]
    followers: tuple[SO101AxisConfig, ...]
    realsense_serials: tuple[str, ...]


def _axis_configs(entries: Sequence[Mapping[str, Any]], *, key: str) -> tuple[SO101AxisConfig, ...]:
    axes = []
    for i, entry in enumerate(entries):
        try:
            axes.append(SO101AxisConfig(port=str(entry["port"]), id=str(entry["id"])))
        except KeyError as e:
            raise ValueError(f"{HARDWARE_CONFIG_YAML}: {key}[{i}] missing required field {e}") from e
    return tuple(axes)


def load_hardware_devices(path: str | Path = HARDWARE_CONFIG_YAML) -> HardwareDevices:
    """Load leader/follower ports+ids and RealSense serials from ``hardware_config.yaml``.

    Uses the same search order as camera extrinsics (env var, cwd, package-adjacent ``src/``);
    see :func:`lerobot_playground.paths.resolve_hardware_config_yaml`.
    """
    resolved = resolve_hardware_config_yaml(path)
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}

    return HardwareDevices(
        leaders=_axis_configs(data.get("leaders") or [], key="leaders"),
        followers=_axis_configs(data.get("followers") or [], key="followers"),
        realsense_serials=tuple(str(s) for s in data.get("realsense_serials") or []),
    )


_DEVICES = load_hardware_devices()
DEFAULT_SO101_LEADERS: tuple[SO101AxisConfig, ...] = _DEVICES.leaders
DEFAULT_SO101_FOLLOWERS: tuple[SO101AxisConfig, ...] = _DEVICES.followers
DEFAULT_ROBOT_CALIBRATION_IDS: tuple[str, ...] = tuple(f.id for f in DEFAULT_SO101_FOLLOWERS)
DEFAULT_REALSENSE_SERIALS: tuple[str, ...] = _DEVICES.realsense_serials


def _validate_axis_sets(
    leaders: Sequence[SO101AxisConfig],
    followers: Sequence[SO101AxisConfig],
    realsense_serials: Sequence[str],
    robot_calibration_ids: Sequence[str],
    robot_calibration_paths: Sequence[str | Path] | None = None,
) -> None:
    if len(leaders) < 1:
        raise ValueError("Need at least one leader (teleop).")
    if len(leaders) != len(followers):
        raise ValueError(
            f"Leaders ({len(leaders)}) and followers ({len(followers)}) must be the same count."
        )
    if len(realsense_serials) < 1:
        raise ValueError("Need at least one RealSense serial.")
    if len(robot_calibration_ids) != len(followers):
        raise ValueError(
            "robot_calibration_ids must have one entry per follower "
            f"(got {len(robot_calibration_ids)} ids for {len(followers)} followers)."
        )
    if robot_calibration_paths is not None and len(robot_calibration_paths) != len(followers):
        raise ValueError(
            "robot_calibration_paths must have one entry per follower "
            f"(got {len(robot_calibration_paths)} paths for {len(followers)} followers)."
        )


@dataclass(frozen=True)
class TeleopSystemConfig:
    """Everything needed to construct :class:`TeleopPointCloudSystem` / :class:`SystemStateViewer`."""

    realsense_serials: tuple[str, ...] = DEFAULT_REALSENSE_SERIALS
    extrinsic_json: str = "extrinsic_calibration.json"
    leaders: tuple[SO101AxisConfig, ...] = DEFAULT_SO101_LEADERS
    followers: tuple[SO101AxisConfig, ...] = DEFAULT_SO101_FOLLOWERS
    recording_name: str = ""
    """Non-empty → write ``recordings/<name>/`` on shutdown."""
    urdf_path: str | None = None
    """``None`` → bundled ``so101_new_calib.urdf`` under package ``calibration/``."""
    robot_calibration_ids: tuple[str, ...] | None = None
    """HF / LeRobot calibration name per follower; ``None`` → each follower's ``id``."""
    robot_calibration_dir: str | Path | None = None
    """Directory containing ``<robot_calibration_id>.json``; ``None`` uses LeRobot defaults."""
    robot_calibration_paths: tuple[str | Path, ...] | None = None
    """Explicit calibration JSON path per follower; overrides ``robot_calibration_dir``."""
    tune: bool = True
    point_size: float = 2.0
    camera_width: int = 848
    camera_height: int = 480
    camera_fps: int = 60
    action_interpolation_duration_s: float = 0.12
    """Seconds to blend from the current command to a new target. ``0`` disables smoothing."""
    action_command_hz: float = 50.0
    """Follower command loop rate when action interpolation is enabled."""
    publish_to_foxglove: bool = True
    """If true, start Foxglove and publish point clouds / transforms."""
    display_point_cloud_viewer: bool = False
    """If true, show full scene + robot clouds in the Open3D point cloud viewer."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "realsense_serials", tuple(self.realsense_serials))
        object.__setattr__(self, "leaders", tuple(self.leaders))
        object.__setattr__(self, "followers", tuple(self.followers))
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_fps <= 0:
            raise ValueError("camera_width, camera_height, and camera_fps must be positive.")
        if self.action_interpolation_duration_s < 0:
            raise ValueError("action_interpolation_duration_s must be >= 0.")
        if self.action_command_hz <= 0:
            raise ValueError("action_command_hz must be positive.")
        rc = self.robot_calibration_ids
        if rc is None or tuple(rc) == DEFAULT_ROBOT_CALIBRATION_IDS:
            rc = tuple(f.id for f in self.followers)
            object.__setattr__(self, "robot_calibration_ids", rc)
        else:
            object.__setattr__(self, "robot_calibration_ids", tuple(rc))
        if self.robot_calibration_paths is not None:
            object.__setattr__(
                self,
                "robot_calibration_paths",
                tuple(Path(p).expanduser() for p in self.robot_calibration_paths),
            )
        _validate_axis_sets(
            self.leaders,
            self.followers,
            self.realsense_serials,
            self.robot_calibration_ids,
            self.robot_calibration_paths,
        )
