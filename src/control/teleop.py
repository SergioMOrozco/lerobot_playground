# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import open3d as o3d
import cv2
from urchin import URDF

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from point_clouds.system_vis import SystemStateViewer

def map_joint_value(raw, raw_range, phys_range):
    return phys_range[0] + (raw - raw_range[0]) / (raw_range[1] - raw_range[0]) * (phys_range[1] - phys_range[0])

class TeleopPointCloudSystem:

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


    def __init__(self,
                 urdf_path,
                 serials,
                 extrinsic_json,
                 intrinsic_json):

        self.robot_urdf = URDF.load(urdf_path)

        #self.robot_1 = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM1", id="bender_follower_arm"))
        #self.robot_2 = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM3", id="clamps_follower_arm"))

        self.teleop_1 = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="bender_leader_arm"))
        self.teleop_2 = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM2", id="clamps_leader_arm"))

        # -------- CAMERA STREAM --------
        self.viewer = SystemStateViewer(serials, extrinsic_json)

        # Cache URDF meshes
        self.meshes_o3d = {}
        self.load_robot_meshes()

    def connect(self):
        print("Connecting devices...")
        #self.robot_1.connect()
        #self.robot_2.connect()
        self.teleop_1.connect()
        self.teleop_2.connect()
        print("Connected.")

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

                self.meshes_o3d[(link.name, mesh_path)] = mesh_o3d

    def compute_robot_points(self, joint_cfg_1, joint_cfg_2):
        """Return sampled + transformed robot points for both arms."""
        robot_pts_1 = []
        robot_pts_2 = []

        for link in self.robot_urdf.links:
            visuals = link.visuals
            if len(visuals) == 0:
                continue

            for visual in visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                key = (link.name, os.path.join("calibration", visual.geometry.mesh.filename))
                mesh_o3d = self.meshes_o3d[key]

                # Sample raw mesh points
                pts_mesh = np.asarray(mesh_o3d.sample_points_uniformly(1000).points)

                # Apply visual origin
                T_vis = visual.origin
                R_vis = T_vis[:3, :3]
                t_vis = T_vis[:3, 3]
                pts_visual = (R_vis @ pts_mesh.T).T + t_vis

                # FK for robot 1
                T1 = self.robot_urdf.link_fk(cfg=joint_cfg_1)[self.robot_urdf.link_map[link.name]]
                R1, t1 = T1[:3, :3], T1[:3, 3]
                robot_pts_1.append((R1 @ pts_visual.T).T + t1)

                # FK for robot 2
                T2 = self.robot_urdf.link_fk(cfg=joint_cfg_2)[self.robot_urdf.link_map[link.name]]
                R2, t2 = T2[:3, :3], T2[:3, 3]
                robot_pts_2.append((R2 @ pts_visual.T).T + t2)

        # Aggregate
        P1 = np.concatenate(robot_pts_1, axis=0)
        P2 = np.concatenate(robot_pts_2, axis=0)
        P2[:,1] += 0.5     # Your original +0.5 shift

        return np.concatenate([P1, P2], axis=0)

    def get_joint_positions(self, action):
        """Map raw teleop values to physical joint angles."""
        return {
            j: map_joint_value(action[f"{j}.pos"], self.RAW_RANGES[j], self.PHYS_RANGES[j])
            for j in self.RAW_RANGES
        }

    def step(self):
        """Perform one full teleop + pointcloud + FK step."""

        # ---- TELEOP 1 ----
        action_1 = self.teleop_1.get_action()
        #self.robot_1.send_action(action_1)
        #joint_cfg_1 = self.get_joint_positions(self.robot_1.get_observation())

        # ---- TELEOP 2 ----
        action_2 = self.teleop_2.get_action()
        #self.robot_2.send_action(action_2)
        #joint_cfg_2 = self.get_joint_positions(self.robot_2.get_observation())

        # ---- ROBOT POINTS ----
        #robot_points = self.compute_robot_points(joint_cfg_1, joint_cfg_2)

        self.viewer.update(action_1, action_2)


    def run(self):
        """Main loop."""
        while True:
            self.step()

if __name__ == "__main__":
    system = TeleopPointCloudSystem(
        urdf_path="calibration/so101_new_calib.urdf",
        #serials=["244622072067", "044322073544"],
        serials=["244622072067"],
        extrinsic_json="extrinsic_calibration.json",
        intrinsic_json="intrinsic_calibration.json"
    )

    system.connect()
    system.run()   # infinite loop