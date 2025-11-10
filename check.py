import numpy as np
bg = np.load("calibration_data/base_to_gripper_transform.npy")
ct = np.load("calibration_data/camera_to_target_transform.npy")

print("Num samples:", bg.shape[0])
print("Gripper translation spread (m):", np.std(bg[:, :3, 3], axis=0))
print("Target translation spread (m):", np.std(ct[:, :3, 3], axis=0))
