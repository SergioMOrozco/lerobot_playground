import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import open3d as o3d

def get_fused_point_cloud(datapoints):
    """Fuses multiple point clouds from different frames into a single point cloud.
    
        Args:
            datapoints (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary contains data for a single camera view. Expected keys
            include 'K' (intrinsic matrix), 'depth' (depth image), 'depth_scale',
            'depth_to_color_extrinsic', 'X_WC' (world-to-camera extrinsic),
            and the key specified by `points_str` (e.g., 'hand_points').
        max_depth (float): The maximum depth value to consider when creating
            the point cloud from a depth image. Points beyond this depth will
            be ignored.

    Returns:
        o3d.geometry.PointCloud: A single Open3D point cloud object containing
            the fused points from all input datapoints, transformed into the
            world coordinate system.
    """

    pc_list = []
    for datapoint in datapoints:

        depth = datapoint["depth"].copy()

        if datapoint["obj_mask"] is not None:
            depth[datapoint["obj_mask"] == 0] = 0.0

        w = depth.shape[1]
        h = depth.shape[0]

        intr = datapoint["color_intrinsics"]  # added by your stream class

        fl_x = intr.fx
        fl_y = intr.fy
        cx = intr.ppx
        cy = intr.ppy
        w = intr.width
        h = intr.height

        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)
        depth_image = o3d.geometry.Image(depth)

        if datapoint["color"] is not None:

            if datapoint["color"].dtype == np.float32:
                img_uint8 = np.array(datapoint["color"] * 255, dtype=np.uint8)
            else:
                img_uint8 = np.array(datapoint["color"])

            color_image = o3d.geometry.Image(img_uint8)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image, convert_rgb_to_intensity=False
            )
            pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, 
                intrinsics
            )

        else:
            pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
                depth_image,
                intrinsics,
                depth_scale=1.0 / datapoint["depth_scale"],
                depth_trunc=datapoint['max_depth'],
            )

        pointcloud.points = o3d.utility.Vector3dVector(pointcloud.points)
        pointcloud.transform(datapoint['X_WC'])

        pc_list.append(pointcloud)

    merged_pc = o3d.geometry.PointCloud()
    for p in pc_list:
        merged_pc += p

    return merged_pc, pc_list 

class MultiRealSenseStream:
    def __init__(self, serial_numbers, extrinsics_file):
        """
        Args:
            serial_numbers (list[str]): e.g. ["0123456789", "9876543210"]
        """
        self.serial_numbers = serial_numbers
        self.pipelines = {}
        self.configs = {}

        self.extrinsics_file = extrinsics_file

        self.get_camera_extrinsics()

        for serial in serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()

            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            pipeline.start(config)

            # Align depth to color
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            self.pipelines[serial] = pipeline
            self.configs[serial] = config

    def get_camera_extrinsics(self):
        """
        Loads and processes camera extrinsic parameters from JSON files.

        Returns:
            tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]

        """

        with open(self.extrinsics_file, "r") as f:
            e = json.load(f)

        extrinsics = {}

        for serial, data in e.items():
            extrinsics[serial] = {
                "X_WC": np.array(data["X_WC"]),
            }

        self.extrinsics = extrinsics

    def get_datapoints(self):
        """
        Returns:
            dict[serial] = {
                "color": np.ndarray(H,W,3),
                "depth": np.ndarray(H,W)
            }
        """
        datapoints = []

        for serial, pipeline in self.pipelines.items():
            frames = pipeline.wait_for_frames()

            aligned_frames = self.align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            profile = color_frame.profile.as_video_stream_profile()
            intr = profile.get_intrinsics()
            color_intrinsics = intr

            color = np.asanyarray(color_frame.get_data())
            aligned_depth = np.asanyarray(aligned_depth_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(aligned_depth, alpha=0.03), cv2.COLORMAP_TURBO
            )

            # Get depth scale from sensor
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            datapoints.append({
                "serial": serial,
                "color": color,
                "depth": aligned_depth,
                "depth_colormap": depth_colormap,
                "depth_scale": depth_scale,
                "max_depth": 10.0,
                "X_WC": self.extrinsics[serial]["X_WC"],
                "color_intrinsics": color_intrinsics,
                "obj_mask": None
            })

        return datapoints

    def stop(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()

#if __name__ == "__main__":
#    #serials = ["244622072067"]
#    serials = ["044322073544"]
#    stream = MultiRealSenseStream(serials, "extrinsic_calibration.json", "intrinsic_calibration.json")
#    pcd_viewer = LivePointCloudViewer()
#
#    while True:
#        datapoints = stream.get_datapoints()
#        fused = get_fused_point_cloud(datapoints)
#        # Convert to numpy
#        pts = np.asarray(fused.points)
#        cols = np.asarray(fused.colors) if fused.has_colors() else None
#        pcd_viewer.update(pts, cols)