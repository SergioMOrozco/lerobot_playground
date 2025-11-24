from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from point_clouds.point_cloud_viewer import LivePointCloudViewer
from urchin import URDF
import os
import numpy as np
import cv2
import open3d as o3d

def map_joint_value(raw, raw_range, phys_range):
    return phys_range[0] + (raw - raw_range[0]) / (raw_range[1] - raw_range[0]) * (phys_range[1] - phys_range[0])

raw_ranges = {
    'shoulder_pan': [-100, 100],
    'shoulder_lift': [-100, 100],
    'elbow_flex': [-100, 100],
    'wrist_flex': [-100, 100],
    'wrist_roll': [-100, 100],
    'gripper': [0, 100] 
}
phys_ranges = {
    'shoulder_pan': [-1.91986, 1.91986],
    'shoulder_lift': [-1.74533, 1.74533],
    'elbow_flex': [-1.69, 1.69],
    'wrist_flex': [-1.65806, 1.65806],
    'wrist_roll': [-2.74385, 2.84121],
    'gripper': [-0.174533, 1.74533]
}

# Load your URDF file
robot_urdf = URDF.load("calibration/so101_new_calib.urdf")

# Define your joint angles in radians
# Example order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

# Compute the forward kinematics to the end effector
#link_name = 'gripper_frame_link'
link_name = 'lower_arm_link'

robot_1_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="bender_follower_arm",
)

teleop_1_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="bender_leader_arm",
)

robot_2_config = SO101FollowerConfig(
    port="/dev/ttyACM3",
    id="clamps_follower_arm",
)

teleop_2_config = SO101LeaderConfig(
    port="/dev/ttyACM2",
    id="clamps_leader_arm",
)

robot_1 = SO101Follower(robot_1_config)
robot_2 = SO101Follower(robot_2_config)
teleop_1_device = SO101Leader(teleop_1_config)
teleop_2_device = SO101Leader(teleop_2_config)
robot_1.connect()
robot_2.connect()
teleop_1_device.connect()
teleop_2_device.connect()


serials = ["244622072067"]
stream = MultiRealSenseStream(serials, "extrinsic_calibration.json", "intrinsic_calibration.json")
pcd_viewer = LivePointCloudViewer()

num = 0

base_to_gripper_transforms = []

while True:
    action = teleop_1_device.get_action()
    robot_1.send_action(action)

    joint_positions = {
        'shoulder_pan': map_joint_value(action['shoulder_pan.pos'], raw_ranges['shoulder_pan'], phys_ranges['shoulder_pan']),
        'shoulder_lift': map_joint_value(action['shoulder_lift.pos'], raw_ranges['shoulder_lift'], phys_ranges['shoulder_lift']),
        'elbow_flex': map_joint_value(action['elbow_flex.pos'], raw_ranges['elbow_flex'], phys_ranges['elbow_flex']),
        'wrist_flex': map_joint_value(action['wrist_flex.pos'], raw_ranges['wrist_flex'], phys_ranges['wrist_flex']),
        'wrist_roll': map_joint_value(action['wrist_roll.pos'], raw_ranges['wrist_roll'], phys_ranges['wrist_roll']),
        'gripper': map_joint_value(action['gripper.pos'], raw_ranges['gripper'], phys_ranges['gripper'])
    }

    #T = robot_urdf.link_fk(cfg=joint_positions)[robot_urdf.link_map[link_name]]

    #action = teleop_2_device.get_action()
    #robot_2.send_action(action)

    #print("Position:", T[:3, 3])

    surface_points_world = []

    for link in robot_urdf.links:

        if len(link.visuals) == 0:
            continue
        visual = link.visuals[0]

        # Must be mesh geometry
        if not hasattr(visual.geometry, "mesh"):
            continue

        # Correct path
        mesh_path = os.path.join("calibration", visual.geometry.mesh.filename)
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        if mesh_o3d.is_empty():
            print("EMPTY MESH:", mesh_path)
            continue

        # Sample points in raw mesh frame
        pcd = mesh_o3d.sample_points_uniformly(500)
        pts_mesh = np.asarray(pcd.points)  # (500,3)

        # ----- APPLY VISUAL ORIGIN TRANSFORM -----
        T_vis = visual.origin  # THIS is your 4×4 transform
        R_vis = T_vis[:3, :3]
        t_vis = T_vis[:3, 3]

        pts_visual = (R_vis @ pts_mesh.T).T + t_vis

        # ----- APPLY LINK FK -----
        T_link = robot_urdf.link_fk(cfg=joint_positions)[robot_urdf.link_map[link.name]]
        R_link = T_link[:3, :3]
        t_link = T_link[:3, 3]

        pts_world = (R_link @ pts_visual.T).T + t_link

        surface_points_world.append(pts_world)

    surface_points_world = np.concatenate(surface_points_world, axis=0)

    datapoints = stream.get_datapoints()
    fused = get_fused_point_cloud(datapoints)

    # Convert to numpy
    pts = np.asarray(fused.points)
    cols = np.asarray(fused.colors) if fused.has_colors() else None
    pcd_viewer.update(pts, surface_points_world, cols)