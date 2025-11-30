# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import cv2
import time
import math
from control.teleop import TeleopPointCloudSystem

def save_transform(json_path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=8)

def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

class TransformEditor:

    def __init__(self, serial):
        self.serial = serial

        self.teleop_system = TeleopPointCloudSystem(
            urdf_path="calibration/so101_new_calib.urdf",
            #urdf_path="calibration/so101_old_calib.urdf",
            serials = [serial],
            extrinsic_json="extrinsic_calibration.json",
            intrinsic_json="intrinsic_calibration.json"
        )
        self.T = self.teleop_system.stream.extrinsics[serial]["X_WC"]

        self.teleop_system.connect()

        # step sizes
        self.trans_step = 0.005     # 5 mm
        self.rot_step = np.deg2rad(2.0)  # 2°

        print("Loaded Transform:\n", self.T)

        cv2.namedWindow("Transform Editor", cv2.WINDOW_NORMAL)
        cv2.imshow("Transform Editor", np.zeros((100, 400, 3), dtype=np.uint8))

    def update_translation(self, axis, sign):
        """axis = 0,1,2 ; sign = +1 or -1"""
        self.T[:3, 3][axis] += sign * self.trans_step

        self.teleop_system.stream.extrinsics[self.serial]["X_WC"] = self.T

    def update_rotation(self, axis, sign):
        """Rotate about X/Y/Z axis by rot_step"""
        R = self.T[:3, :3]
        step = sign * self.rot_step

        if axis == 0:
            R_new = R @ rot_x(step)
        elif axis == 1:
            R_new = R @ rot_y(step)
        elif axis == 2:
            R_new = R @ rot_z(step)

        self.T[:3, :3] = R_new

        self.teleop_system.stream.extrinsics[self.serial]["X_WC"] = self.T

    def run(self):
        print("\n------ CONTROL KEYS (OpenCV) ------")
        print("q/w : -/+ X translation")
        print("a/s : -/+ Y translation")
        print("z/x : -/+ Z translation")
        print("t/y : -/+ roll   (rotate X)")
        print("g/h : -/+ pitch  (rotate Y)")
        print("b/n : -/+ yaw    (rotate Z)")
        print("p   : quit")
        print("-----------------------------------\n")

        while True:

            self.teleop_system.step()

            key = cv2.waitKey(1) & 0xFF
            if key == 255:   # no key pressed
                continue

            k = chr(key)

            if k == "p":
                print("Exiting transform editor.")
                break

            # TRANSLATION
            if k == "q": self.update_translation(0, -1)
            if k == "w": self.update_translation(0, +1)

            if k == "a": self.update_translation(1, -1)
            if k == "s": self.update_translation(1, +1)

            if k == "z": self.update_translation(2, -1)
            if k == "x": self.update_translation(2, +1)

            # ROTATION
            if k == "t": self.update_rotation(0, -1)
            if k == "y": self.update_rotation(0, +1)

            if k == "g": self.update_rotation(1, -1)
            if k == "h": self.update_rotation(1, +1)

            if k == "b": self.update_rotation(2, -1)
            if k == "n": self.update_rotation(2, +1)

        print(self.T)

editor = TransformEditor(
    #serial="244622072067"
    serial="044322073544"
)

editor.run()