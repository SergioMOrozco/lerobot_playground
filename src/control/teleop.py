# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import sys
import argparse

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

    def step(self):
        """Perform one full teleop + pointcloud + FK step."""

        # ---- TELEOP 1 ----
        action_1 = self.teleop_1.get_action()

        # ---- TELEOP 2 ----
        action_2 = self.teleop_2.get_action()

        self.viewer.update(action_1, action_2)


    def run(self):
        """Main loop."""
        while True:
        #for i in range(100):

            if self.viewer.quit:
                break

            self.step()

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

    system.connect()
    system.run()   # infinite loop