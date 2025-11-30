import foxglove
import time
import logging
import numpy as np
import math
import cv2
import inspect
#import datetime

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader

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
from point_clouds.point_cloud_viewer import LivePointCloudViewer
from foxglove.schemas import Pose, Vector3, Quaternion

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

def map_joint_value(raw, raw_range, phys_range):
    return phys_range[0] + (raw - raw_range[0]) / (raw_range[1] - raw_range[0]) * (phys_range[1] - phys_range[0])

def get_joint_positions(action):
    """Map raw teleop values to physical joint angles."""
    return {
        j: map_joint_value(action[f"{j}.pos"], RAW_RANGES[j], PHYS_RANGES[j])
        for j in RAW_RANGES
    }

from urchin import URDF

# Hack to ensure np.float works with ancient urdfpy version
if not hasattr(np, 'float'):
    np.float = float

WORLD_FRAME_ID = "world"
BASE_FRAME_ID = "base"
RATE_HZ = 30.0
URDF_FILE = "calibration/so101_new_calib.urdf"
WRIST_CAM_ID = 0
ENV_CAM_ID = 4

def rot_matrix_to_quat(R):
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

def main():
    foxglove.set_log_level(logging.INFO)

    print(f"Loading URDF from {URDF_FILE} ...")
    robot = URDF.load(URDF_FILE)

    # Start the Foxglove server
    server = foxglove.start_server()

    serials=["244622072067"]
    extrinsic_json="extrinsic_calibration.json"
    intrinsic_json="intrinsic_calibration.json"

    stream = MultiRealSenseStream(serials, extrinsic_json, intrinsic_json)

    print("Open Foxglove Studio and connect to ws://localhost:8765")

    config = SO101FollowerConfig(port="/dev/ttyACM1", id="bender_follower_arm")

    follower = SO101Follower(config)
    follower.connect(calibrate=False)

    leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="bender_leader_arm"))
    leader.connect(calibrate=False)

    if not follower.is_connected:
        print("Failed to connect to SO-100 Follower arm. Please check the connection.")
        return
    print("SO-100 Follower arm connected successfully.")
    follower.bus.disable_torque() # Disable torque to be able to move the arm freely

    # Define initial joint positions (all zeros for now)
    joint_positions = {}
    joint_positions_offsets = {}
    for joint in robot.joints:
        joint_positions[joint.name] = 0.0
        joint_positions_offsets[joint.name] = 0.0

    print(f"Available joints: {list(joint_positions.keys())}")

    cv2.namedWindow("Transform Editor", cv2.WINDOW_NORMAL)
    cv2.imshow("Transform Editor", np.zeros((100, 400, 3), dtype=np.uint8))

    try:
        while True:

            datapoints = stream.get_datapoints()
            fused_pcd = get_fused_point_cloud(datapoints)

            action = leader.get_action()
            follower.send_action(action)

            # Read actual joint angles from follower (in degrees)
            obs = follower.get_observation()

            #TODO: The URDF 0 position does not match the calibrated arm 0 position
            # as described in the lerobot documentation.

            joint_positions = get_joint_positions(obs)

            for joint in robot.joints:
                if joint.name in joint_positions and joint.name in joint_positions_offsets:
                    joint_positions[joint.name] += joint_positions_offsets[joint.name]

            key = cv2.waitKey(1) & 0xFF

            k = chr(key)

            if k == "p":
                print("Exiting transform editor.")
                break

            # TRANSLATION
            if k == "1": joint_positions_offsets['shoulder_pan'] += 0.01 
            if k == "q": joint_positions_offsets['shoulder_pan'] -= 0.01 
            if k == "2": joint_positions_offsets['shoulder_lift'] += 0.01 
            if k == "w": joint_positions_offsets['shoulder_lift'] -= 0.01 
            if k == "3": joint_positions_offsets['elbow_flex'] += 0.01 
            if k == "e": joint_positions_offsets['elbow_flex'] -= 0.01 
            if k == "4": joint_positions_offsets['wrist_flex'] += 0.01 
            if k == "r": joint_positions_offsets['wrist_flex'] -= 0.01 
            if k == "5": joint_positions_offsets['wrist_roll'] += 0.01 
            if k == "t": joint_positions_offsets['wrist_roll'] -= 0.01 

            # Something is wrong with the gripper. There is some nonlinearity going on I believe
            if k == "6": joint_positions_offsets['gripper'] += 0.01 
            if k == "y": joint_positions_offsets['gripper'] -= 0.01 

            print(joint_positions_offsets)

            # Compute forward kinematics with updated joint positions
            fk_poses = robot.link_fk(cfg=joint_positions)

            transforms = []
            # World -> Base
            transforms.append(
                FrameTransform(
                    parent_frame_id=WORLD_FRAME_ID,
                    child_frame_id=BASE_FRAME_ID,
                    translation=Vector3(x=0.0, y=0.0, z=0.0),
                    rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )
            )
            # Per-joint transforms
            for joint in robot.joints:
                parent_link = joint.parent
                child_link = joint.child
                T_parent = fk_poses[robot.link_map[parent_link]]
                T_child = fk_poses[robot.link_map[child_link]]
                # Local transform from parent->child
                T_local = np.linalg.inv(T_parent) @ T_child
                trans = T_local[:3, 3]
                quat = rot_matrix_to_quat(T_local[:3, :3])
                transforms.append(
                    FrameTransform(
                        parent_frame_id=parent_link,
                        child_frame_id=child_link,
                        translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                        rotation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
                    )
                )

            foxglove.log(
                "/tf",
                FrameTransforms(transforms=transforms)
            )

            pcd_msg = foxglove_pointcloud_from_numpy(np.asarray(fused_pcd.points))
            foxglove.log("/fused_point_cloud", pcd_msg)

            time.sleep(1.0 / RATE_HZ)

    except KeyboardInterrupt:
        print("\nShutting down Foxglove viewer...")
        server.stop()
        follower.disconnect()
        camera.disconnect()
        camera2.disconnect()

if __name__ == "__main__":
    main()

