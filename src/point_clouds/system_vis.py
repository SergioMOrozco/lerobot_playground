import foxglove
import logging
import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import open3d as o3d
import numpy as np

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
from foxglove.schemas import Pose, Vector3, Quaternion
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

def foxglove_pointcloud_from_numpy(points: np.ndarray):
    # points = (N,3)
    data = points.astype(np.float32).tobytes()

    # Foxglove expects PackedElementField(name, offset_bytes, datatype)
    fields=[
        PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
    ]

    identity_pose = Pose(
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    )

    return PointCloud(
        timestamp=None,        # optional
        frame_id="base_link",
        pose=identity_pose,
        point_stride=12,       # 3 floats × 4 bytes
        fields=fields,
        data=data
    )

class SystemStateViewer:
    def __init__(self, serials, extrinsic_json, point_size=2.0):

        self.stream = MultiRealSenseStream(serials, extrinsic_json)
        self.robot_1 = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM1", id="bender_follower_arm"))
        self.robot_2 = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM3", id="clamps_follower_arm"))

        print("Connecting robots...")
        self.robot_1.connect()
        self.robot_2.connect()
        print("Connected.")

        self.robot_state = RobotState("calibration/so101_new_calib.urdf")

        foxglove.set_log_level(logging.INFO)

        # Start the Foxglove server
        server = foxglove.start_server()

    def update(self, action_1, action_2):

        self.robot_1.send_action(action_1)
        self.robot_2.send_action(action_2)

        obs = self.robot_1.get_observation()

        transforms, robot_pcd_np = self.robot_state.get_transforms(obs)
        robot_pcd_msg = foxglove_pointcloud_from_numpy(np.asarray(robot_pcd_np))

        datapoints = self.stream.get_datapoints()
        scene_pcd = get_fused_point_cloud(datapoints)

        pts = np.asarray(scene_pcd.points)
        cols = np.asarray(scene_pcd.colors) if scene_pcd.has_colors() else None
        scene_pcd_msg = foxglove_pointcloud_from_numpy(np.asarray(scene_pcd.points))

        foxglove.log("/scene_pcd", scene_pcd_msg)
        foxglove.log("/robot_pcd", robot_pcd_msg)
        foxglove.log(
            "/tf",
            FrameTransforms(transforms=transforms)
        )

class RobotState:
    RAW_RANGES = {
        'shoulder_pan': [-100, 100],
        'shoulder_lift': [-100, 100],
        'elbow_flex': [-100, 100],
        'wrist_flex': [-100, 100],
        'wrist_roll': [-100, 100],
        'gripper': [0, 100] 
    }

    PHYS_RANGES = {
        'shoulder_pan': [-1.91986, 1.91986],
        'shoulder_lift': [-1.74533, 1.74533],
        'elbow_flex': [-1.69, 1.69],
        'wrist_flex': [-1.65806, 1.65806],
        'wrist_roll': [-2.74385, 2.84121],
        'gripper': [-0.174533, 1.74533]
    }

    def __init__(self, urdf_path):
        self.robot_urdf = URDF.load(urdf_path)

        # Cache URDF meshes
        self.robot_meshes_o3d = {}
        self.load_robot_meshes()


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

    def get_joint_positions(self, obs):
        """Map raw teleop values to physical joint angles."""
        return {
            j: self.map_joint_value(obs[f"{j}.pos"], self.RAW_RANGES[j], self.PHYS_RANGES[j])
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

    def sample_robot_points(self, obs):
        """Return sampled + transformed robot points for both arms."""
        robot_pts = []

        joint_positions = self.get_joint_positions(obs)

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

    #def sample_robot_points(self, obs, points_per_link=200):
    #    joint_positions = self.get_joint_positions(obs)
    #    fk_poses = self.robot_urdf.link_fk(cfg=joint_positions)

    #    all_points = []

    #    for link in self.robot_urdf.links:
    #        T = fk_poses[self.robot_urdf.link_map[link.name]]

    #        for vis in link.visuals:
    #            geom = vis.geometry

    #            # Only mesh geometry
    #            if hasattr(geom, "mesh"):
    #                V = geom.mesh.vertices
    #                
    #                # Random sampling
    #                if len(V) > points_per_link:
    #                    idx = np.random.choice(len(V), size=points_per_link, replace=False)
    #                    V = V[idx]

    #                # Transform to world
    #                V_h = np.hstack([V, np.ones((V.shape[0], 1))])
    #                V_world = (T @ V_h.T).T[:, :3]

    #                all_points.append(V_world)

    #    if len(all_points) == 0:
    #        return np.empty((0, 3))

    #    return np.vstack(all_points)
    

    def get_transforms(self, obs):

        transforms = []

        joint_positions = self.get_joint_positions(obs)

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

        return transforms, self.sample_robot_points(obs)