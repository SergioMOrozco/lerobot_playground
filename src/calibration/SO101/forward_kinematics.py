from urdfpy import URDF
import numpy as np

# Load your URDF file
robot = URDF.load("so101_new_calib.urdf")

# Define your joint angles in radians
# Example order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
joint_positions = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.5,
    'elbow_flex': -0.7,
    'wrist_flex': 0.2,
    'wrist_roll': 1.0,
    'gripper': 0.0
}

# Compute the forward kinematics to the end effector
link_name = 'gripper_frame_link'
T = robot.link_fk(cfg=joint_positions)[robot.link_map[link_name]]

print("End Effector Pose:")
print(T)  # 4x4 transformation matrix (position + orientation)
print("Position:", T[:3, 3])
