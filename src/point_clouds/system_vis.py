import foxglove
import logging
import os
import sys
import json
import shutil

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import open3d as o3d
import numpy as np
import imageio
import cv2

from urchin import URDF
from foxglove.schemas import (
    RawImage,
    FrameTransforms,
    FrameTransform,
    Vector3,
    Quaternion,
)
from foxglove.channels import RawImageChannel, PointCloudChannel
from foxglove.schemas import PointCloud, PackedElementField, PackedElementFieldNumericType
from point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from point_clouds.tuner import StateTuner 
from foxglove.schemas import Pose, Vector3, Quaternion
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

def foxglove_pointcloud_from_numpy(points: np.ndarray, colors=None):

    N = points.shape[0]
    assert points.shape[1] == 3

    # Default alpha = 255
    if colors is None:
        colors = np.zeros((N, 3), dtype=np.uint8)

    # Ensure uint8
    colors = colors.astype(np.uint8)

    # Add alpha channel (uint8 = 255)
    a = np.full((N,1), 255, dtype=np.uint8)

    # Build structured array matching Foxglove's fields
    structured = np.zeros(N, dtype=[
        ("x", "float32"),
        ("y", "float32"),
        ("z", "float32"),
        ("b", "uint8"),
        ("g", "uint8"),
        ("r", "uint8"),
        ("a", "uint8"),
    ])

    structured["x"] = points[:,0]
    structured["y"] = points[:,1]
    structured["z"] = points[:,2]

    structured["r"] = colors[:,0]
    structured["g"] = colors[:,1]
    structured["b"] = colors[:,2]
    structured["a"] = 255

    data = structured.tobytes()

    fields = [
        PackedElementField(name="x", offset=0,  type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="y", offset=4,  type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="z", offset=8,  type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="red", offset=12, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="green", offset=13, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="blue", offset=14, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="alpha", offset=15, type=PackedElementFieldNumericType.Uint8),
    ]

    identity_pose = Pose(
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    )

    return PointCloud(
        timestamp=None,
        frame_id="base_link",
        pose=identity_pose,
        point_stride=16,
        fields=fields,
        data=data
    )

class SystemStateViewer:
    def __init__(self, serials, extrinsic_json, recording_name, point_size=2.0, tune=True):

        self.stream = MultiRealSenseStream(serials, extrinsic_json)
        self.robot_1 = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM1", id="bender_follower_arm"))
        self.robot_2 = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM3", id="clamps_follower_arm"))

        self.recording_name = recording_name

        if self.recording_name != '':
            self.record = True
        else:
            self.record = False

        self.quit=False

        self.prev_tuned_state = None

        if tune:
            self.state_tuner = StateTuner(
                [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper"
                ]
            )
            self.state_tuner.start()

        print("Connecting robots...")
        self.robot_1.connect()
        self.robot_2.connect()
        print("Connected.")

        self.robot_state = RobotState("calibration/so101_new_calib.urdf", "bender_follower_arm")

        foxglove.set_log_level(logging.INFO)

        # Start the Foxglove server
        server = foxglove.start_server()

        self.serials = serials
        self.images = {}
        self.depths = {}

        for serial in serials:
            self.images[serial] = []
            self.depths[serial] = []

    def rpy_to_matrix(self, rpy):

        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_x = np.array([[1, 0, 0],
                        [0, cr, -sr],
                        [0, sr,  cr]])

        R_y = np.array([[ cp, 0, sp],
                        [  0, 1,  0],
                        [-sp, 0, cp]])

        R_z = np.array([[cy, -sy, 0],
                        [sy,  cy, 0],
                        [  0,   0, 1]])

        return R_z @ R_y @ R_x   # yaw * pitch * roll

    def export_fine_tuning(self):
        datapoints = self.stream.get_datapoints()

        extrinsics = {}

        for datapoint in datapoints:

            matrix_list = datapoint['X_WC'].tolist()

            extrinsics[datapoint['serial']] = {
                "X_WC": matrix_list
            }

        with open("extrinsic_calibration.json", "w") as f:
            json.dump(extrinsics, f, indent=8)

    def tune_state(self):
        tuned_state = self.state_tuner.get_state()

        if self.prev_tuned_state is not None:
            self.stream.extrinsics["244622072067"]['X_WC'][:3, 3] -= self.prev_tuned_state['cam1_pos']
            self.stream.extrinsics['244622072067']['X_WC'][:3, :3] @= self.rpy_to_matrix(self.prev_tuned_state['cam1_rot']).T

        self.stream.extrinsics["244622072067"]['X_WC'][:3, 3] += tuned_state['cam1_pos']
        self.stream.extrinsics["244622072067"]['X_WC'][:3, :3] @=  self.rpy_to_matrix(tuned_state['cam1_rot'])

        if self.prev_tuned_state is not None:
            self.stream.extrinsics["044322073544"]['X_WC'][:3, 3] -= self.prev_tuned_state['cam2_pos']
            self.stream.extrinsics["044322073544"]['X_WC'][:3, :3] @= self.rpy_to_matrix(self.prev_tuned_state['cam2_rot']).T

        self.stream.extrinsics["044322073544"]['X_WC'][:3, 3] += tuned_state['cam2_pos']
        self.stream.extrinsics["044322073544"]['X_WC'][:3, :3] @=  self.rpy_to_matrix(tuned_state['cam2_rot'])

        self.tuned_joint_offsets = tuned_state['joint_offsets']
        self.prev_tuned_state = tuned_state

    def update(self, action_1, action_2):

        if self.state_tuner.quit is True:
            self.quit = True

        self.tune_state()

        self.robot_1.send_action(action_1)
        self.robot_2.send_action(action_2)

        obs = self.robot_1.get_observation()

        transforms, robot_pcd_np = self.robot_state.get_transforms(obs, self.tuned_joint_offsets)
        robot_pcd_np = robot_pcd_np.reshape(robot_pcd_np.shape[0] * robot_pcd_np.shape[1], 3)
        robot_pcd_msg = foxglove_pointcloud_from_numpy(np.asarray(robot_pcd_np))

        datapoints = self.stream.get_datapoints()

        if self.record:
            for datapoint in datapoints:
                self.images[datapoint['serial']].append(np.array(datapoint['color']))
                self.depths[datapoint['serial']].append(np.array(datapoint['depth']))

        scene_pcd, pcd_list = get_fused_point_cloud(datapoints)

        for idx, pcd in enumerate(pcd_list):
            pts = np.asarray(pcd.points, dtype=np.float32)
            cols = np.asarray(np.array(pcd.colors) * 255, dtype=np.uint8) if pcd.has_colors() else None
            pcd_msg = foxglove_pointcloud_from_numpy(pts, cols)

            foxglove.log(f"/pcd_{idx}", pcd_msg)

        foxglove.log("/robot_pcd", robot_pcd_msg)
        foxglove.log(
            "/tf",
            FrameTransforms(transforms=transforms)
        )

    def close(self):
        if self.record:

            recording_dir = f"recordings/{self.recording_name}"

            # remove task directory if it exists
            if os.path.exists(recording_dir):
                shutil.rmtree(recording_dir)

            os.makedirs(recording_dir)

            for serial in self.serials:

                serial_dir = os.path.join(recording_dir, f"{serial}" )

                # remove task directory if it exists
                if os.path.exists(serial_dir):
                    shutil.rmtree(serial_dir)

                os.makedirs(serial_dir)

                frames_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images[serial]]
                frames_depth = self.depths[serial]

                imageio.mimsave(
                    os.path.join(serial_dir, "rgb.mp4"),
                    frames_rgb,
                    fps=30,
                    codec="libx264"
                )

                np.savez_compressed(os.path.join(serial_dir, "depth.npz"), depth=np.array(frames_depth))

        print("export fine tuning? y/n")
        answer = input()

        if answer == "y":
            self.export_fine_tuning()

        self.stream.stop()

class RobotState:
    RAW_RANGES = {
        'shoulder_pan': [-100, 100],
        'shoulder_lift': [-100, 100],
        'elbow_flex': [-100, 100],
        'wrist_flex': [-100, 100],
        'wrist_roll': [-100, 100],
        'gripper': [0, 100] 
    }

    def __init__(self, urdf_path, id):
        self.robot_urdf = URDF.load(urdf_path)

        robot_calibration_path = f"/home/sorozco0612/.cache/huggingface/lerobot/calibration/robots/so101_follower/{id}.json"

        with open(robot_calibration_path, "r") as f:
            calib = json.load(f)

        self.PHYS_RANGES = self.compute_phys_ranges(calib)

        # Cache URDF meshes
        self.robot_meshes_o3d = {}
        self.load_robot_meshes()

    def ticks_to_radians(self, raw, homing_offset):
        TICKS_PER_REV = 4096
        return (raw + homing_offset) * (2 * np.pi / TICKS_PER_REV)
        #return (raw) * (2 * np.pi / TICKS_PER_REV)

    def compute_phys_ranges(self, calib_dict):
        phys_ranges = {}

        for joint, data in calib_dict.items():
            range_min = data["range_min"]
            range_max = data["range_max"]
            offset    = data["homing_offset"]

            lo = self.ticks_to_radians(range_min, offset)
            hi = self.ticks_to_radians(range_max, offset)

            phys_ranges[joint] = [float(lo), float(hi)]

        return phys_ranges

    def rot_matrix_to_quat(self, R):
        """
        Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float64)

    def map_joint_value(self, raw, raw_range, phys_range):
        return phys_range[0] + (raw - raw_range[0]) / (raw_range[1] - raw_range[0]) * (phys_range[1] - phys_range[0])

    def get_joint_positions(self, obs, tuned_joint_offsets):
        """Map raw teleop values to physical joint angles."""

        if tuned_joint_offsets is None:
            return {
                j: self.map_joint_value(obs[f"{j}.pos"], self.RAW_RANGES[j], self.PHYS_RANGES[j])
                for j in self.RAW_RANGES
            }
        else:
            return {
                j: self.map_joint_value(obs[f"{j}.pos"], self.RAW_RANGES[j], self.PHYS_RANGES[j]) + tuned_joint_offsets[f"{j}"]
                for j in self.RAW_RANGES
            }

    def load_robot_meshes(self):
        """Load and sample all robot meshes upfront."""
        for link in self.robot_urdf.links:
            for visual in link.visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                mesh_path = os.path.join("calibration", visual.geometry.mesh.filename)
                mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)

                if mesh_o3d.is_empty():
                    print(f"[WARN] Empty mesh: {mesh_path}")
                    continue

                self.robot_meshes_o3d[(link.name, mesh_path)] = mesh_o3d

    def sample_robot_points(self, obs, tuned_joint_offsets):
        """Return sampled + transformed robot points for both arms."""
        robot_pts = []

        joint_positions = self.get_joint_positions(obs, tuned_joint_offsets)

        for link in self.robot_urdf.links:
            visuals = link.visuals
            if len(visuals) == 0:
                continue

            for visual in visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                key = (link.name, os.path.join("calibration", visual.geometry.mesh.filename))
                mesh_o3d = self.robot_meshes_o3d[key]

                # Sample raw mesh points
                pts_mesh = np.asarray(mesh_o3d.sample_points_uniformly(1000).points)

                # Apply visual origin
                T_vis = visual.origin
                R_vis = T_vis[:3, :3]
                t_vis = T_vis[:3, 3]
                pts_visual = (R_vis @ pts_mesh.T).T + t_vis

                # FK for robot
                T1 = self.robot_urdf.link_fk(cfg=joint_positions)[self.robot_urdf.link_map[link.name]]
                R1, t1 = T1[:3, :3], T1[:3, 3]
                robot_pts.append((R1 @ pts_visual.T).T + t1)

        return np.array(robot_pts)

    def get_transforms(self, obs, tuned_joint_offsets):

        transforms = []

        joint_positions = self.get_joint_positions(obs, tuned_joint_offsets)

        # Compute forward kinematics with updated joint positions
        fk_poses = self.robot_urdf.link_fk(cfg=joint_positions)

        # World -> Base
        transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="base",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        )

        for joint in self.robot_urdf.joints:
            parent_link = joint.parent
            child_link = joint.child
            T_parent = fk_poses[self.robot_urdf.link_map[parent_link]]
            T_child = fk_poses[self.robot_urdf.link_map[child_link]]

            # Local transform from parent->child
            T_local = np.linalg.inv(T_parent) @ T_child
            trans = T_local[:3, 3]
            quat = self.rot_matrix_to_quat(T_local[:3, :3])
            transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                    rotation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
                )
            )

        return transforms, self.sample_robot_points(obs, tuned_joint_offsets)