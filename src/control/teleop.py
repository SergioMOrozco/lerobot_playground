# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import sys
import argparse
import cv2
import numpy as np

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from point_clouds.system_vis import SystemStateViewer

class TeleopPointCloudSystem:

    def __init__(self,
                 serials,
                 extrinsic_json,
                 recording_name):

        self.teleop_1 = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="bender_leader_arm"))
        self.teleop_2 = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM2", id="clamps_leader_arm"))

        self.viewer = SystemStateViewer(serials, extrinsic_json, recording_name)

    def connect(self):
        print("Connecting devices...")
        self.teleop_1.connect()
        self.teleop_2.connect()
        print("Connected.")

    def run(self, calibrate= False, serial_calibrate=None):
        """Main loop."""

        num = 0

        base_to_gripper_transforms = []

        while True:

            if self.viewer.quit:
                break

            # ---- TELEOP 1 ----
            action_1 = self.teleop_1.get_action()

            # ---- TELEOP 2 ----
            action_2 = self.teleop_2.get_action()

            self.viewer.update(action_1, action_2)

            if calibrate and self.viewer.state_tuner.capture:

                self.viewer.state_tuner.capture = False

                base2gripper_T = self.viewer.robot_state.get_eef_pos(self.viewer.robot_1.get_observation(), self.viewer.tuned_joint_offsets)
                datapoints = self.viewer.stream.get_datapoints()

                for datapoint in datapoints:
                    if datapoint['serial'] == serial_calibrate:

                        if not os.path.exists(f"calibration/calibration_data/images/{datapoint['serial']}"):
                            os.makedirs(f"calibration/calibration_data/images/{datapoint['serial']}")

                        cv2.imwrite(f"calibration/calibration_data/images/{datapoint['serial']}/{num}.png", datapoint['color'])
                        print("image saved!")

                num += 1

                base_to_gripper_transforms.append(base2gripper_T)

        if calibrate:
            np.save('calibration/calibration_data/base_to_gripper_transform.npy', np.array(base_to_gripper_transforms))

        self.viewer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--recording_name", type=str, default="", help="which config to load"
    )

    args = parser.parse_args()

    system = TeleopPointCloudSystem(
        serials=["244622072067", "044322073544"],
        extrinsic_json="extrinsic_calibration.json",
        recording_name=args.recording_name
    )

    calibrate = False
    serial_calibrate = "244622072067"

    system.connect()
    system.run(calibrate, serial_calibrate)   # infinite loop