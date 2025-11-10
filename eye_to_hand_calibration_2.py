import numpy as np
import cv2

def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):

    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)

        # change parameters values
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # calibrate
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )

    return R, t

base_to_gripper_transform = np.load('calibration_data/base_to_gripper_transform.npy')
camera_to_target_transform = np.load('calibration_data/camera_to_target_transform.npy')

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

R_base2gripper = []
t_base2gripper = []
R_cam2target = []
t_cam2target = []

for T_bg, T_ct in zip(base_to_gripper_transform, camera_to_target_transform):
    # Invert transforms

    T_gb = T_bg # gripper -> base
    T_tc = T_ct # target -> camera

    # Extract R, t
    R_gripper2base.append(T_gb[:3, :3])
    t_gripper2base.append(T_gb[:3, 3])
    R_target2cam.append(T_tc[:3, :3])
    t_target2cam.append(T_tc[:3, 3])

    # Extract R, t
    R_base2gripper.append(T_bg[:3, :3])
    t_base2gripper.append(T_bg[:3, 3])
    R_cam2target.append(T_ct[:3, :3])
    t_cam2target.append(T_ct[:3, 3])

R_camera2base, t_camera2base = calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=False)

T_camera2base = np.eye(4)
T_camera2base[:3, :3] = R_camera2base
T_camera2base[:3, 3] = t_camera2base.squeeze()

T_base2camera = np.linalg.inv(T_camera2base)

print(T_camera2base)

#for T_base2gripper, T_cam2target in zip(base_to_gripper_transform,camera_to_target_transform):
#
#    T_gripper2base = T_base2gripper
#    T_target2cam = T_cam2target
#
#    T_base2cam = T_base2gripper @ T_gripper2target @ T_target2cam
#
#    print(T_base2cam[:3,3])
