import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import open3d as o3d

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

from sklearn.neighbors import NearestNeighbors

from point_clouds.point_cloud_viewer import LivePointCloudViewer


def _crop_point_cloud_to_aabb(pcd: o3d.geometry.PointCloud, lo: np.ndarray, hi: np.ndarray):
    """Keep only points inside the axis-aligned box [lo, hi] (world frame), inclusive."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    if not mask.any():
        out = o3d.geometry.PointCloud()
        return out
    idx = np.nonzero(mask)[0]
    return pcd.select_by_index(idx)


def _prepare_robot_reference_tree(
    reference_xyz: np.ndarray,
    radius: float,
    *,
    max_ref_points: int = 4096,
    ref_voxel_m=None,
):
    """Voxel-downsample + cap robot samples, then build a KD-tree (fast queries)."""
    ref = np.asarray(reference_xyz, dtype=np.float64).reshape(-1, 3)
    ref = ref[np.isfinite(ref).all(axis=1)]
    if ref.shape[0] == 0:
        return None

    voxel = ref_voxel_m if ref_voxel_m is not None else max(radius * 0.25, 0.005)
    rpcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref))
    rpcd = rpcd.voxel_down_sample(float(voxel))
    ref = np.asarray(rpcd.points, dtype=np.float64)
    if ref.shape[0] == 0:
        return None

    if ref.shape[0] > max_ref_points:
        sel = np.random.default_rng().choice(ref.shape[0], size=max_ref_points, replace=False)
        ref = ref[sel]

    if cKDTree is not None:
        return cKDTree(ref)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(ref)
    return nn


def _nearest_dists_to_tree(tree, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64)
    if cKDTree is not None and isinstance(tree, cKDTree):
        try:
            dists, _ = tree.query(pts, k=1, workers=-1)
        except TypeError:
            dists, _ = tree.query(pts, k=1)
        return np.asarray(dists, dtype=np.float64).reshape(-1)
    dists, _ = tree.kneighbors(pts)
    return dists[:, 0]


def _remove_points_near_reference_tree(
    pcd: o3d.geometry.PointCloud,
    tree,
    radius: float,
) -> o3d.geometry.PointCloud:
    """Drop points with nearest-neighbor distance to robot samples <= radius."""
    if tree is None or radius <= 0:
        return pcd
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd
    dists = _nearest_dists_to_tree(tree, pts)
    keep = dists > radius
    if not keep.any():
        return o3d.geometry.PointCloud()
    return pcd.select_by_index(np.nonzero(keep)[0])


def get_fused_point_cloud(
    datapoints,
    bbox_3d=None,
    robot_points_world=None,
    robot_exclude_radius=None,
    robot_ref_max_points=4096,
    robot_ref_voxel_m=None,
    projection_stride=1,
    scene_voxel_m=None,
):
    """Fuse per-camera depth (and optional color) into one world-frame point cloud.

    Most wall time is usually **depth→points** (pixel count) and **scene NN/bbox**
    work, not ``robot_ref_max_points`` (that only shrinks the robot KD-tree).

    Args:
        datapoints: List of dicts per camera with ``depth``, ``color_intrinsics``,
            ``depth_scale``, ``max_depth`` (depth-only path), ``X_WC`` (4x4 world
            from camera), optional ``color``, optional ``obj_mask``.
        bbox_3d: Optional axis-aligned box in **world** coordinates after each
            cloud is transformed by ``X_WC``. Shape ``(3, 2)``: row ``i`` is
            ``[min, max]`` for ``(x, y, z)``. Points outside are removed from the
            merged cloud and each entry in ``pc_list``. Same layout as
            ``get_bounding_box()`` in ``postprocess.py``. Pass ``None`` to disable.
        robot_points_world: Optional ``(N, 3)`` robot surface samples in the **same
            world frame** as fused camera points (e.g. sampled URDF mesh). Fused
            points within ``robot_exclude_radius`` of any sample are removed.
        robot_exclude_radius: Distance threshold in meters. If ``None`` or ``<= 0``,
            robot filtering is disabled. If ``robot_points_world`` is set but this
            is ``None``, defaults to ``0.03``.
        robot_ref_max_points: Max robot samples after voxel downsample (speed vs coverage).
        robot_ref_voxel_m: Voxel size (m) for robot cloud decimation; ``None`` uses
            ``max(radius * 0.25, 0.005)``.
        projection_stride: Integer >= 1. ``2`` halves depth/color resolution before
            back-projection (~4× fewer points). Intrinsics are scaled accordingly.
        scene_voxel_m: If set, voxel-downsample each camera cloud in **world** space
            after transform and before bbox/robot (large speedup).

    Returns:
        (merged_pc, pc_list): One merge at the end; per-camera clouds stay consistent
        with ``merged_pc``.
    """

    if projection_stride < 1:
        raise ValueError("projection_stride must be >= 1")
    st = int(projection_stride)

    pc_list = []
    for datapoint in datapoints:

        depth = datapoint["depth"].copy()
        depth = np.ascontiguousarray(depth.astype(np.float32))

        if datapoint["obj_mask"] is not None:
            depth[datapoint["obj_mask"] == 0] = 0.0

        intr = datapoint["color_intrinsics"]  # added by your stream class

        fl_x = intr.fx / st
        fl_y = intr.fy / st
        cx = intr.ppx / st
        cy = intr.ppy / st
        if st > 1:
            depth = depth[::st, ::st]
        depth = np.ascontiguousarray(depth)

        w = int(depth.shape[1])
        h = int(depth.shape[0])

        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)
        depth_image = o3d.geometry.Image(depth)

        if datapoint["color"] is not None:

            color_arr = datapoint["color"]
            if st > 1:
                color_arr = color_arr[::st, ::st]

            if color_arr.dtype == np.float32:
                img_uint8 = np.ascontiguousarray(np.array(color_arr * 255, dtype=np.uint8))
            else:
                img_uint8 = np.ascontiguousarray(np.asarray(color_arr, dtype=np.uint8))

            color_image = o3d.geometry.Image(img_uint8)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image, convert_rgb_to_intensity=False
            )
            pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsics,
            )

        else:
            pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
                depth_image,
                intrinsics,
                depth_scale=1.0 / datapoint["depth_scale"],
                depth_trunc=datapoint["max_depth"],
            )

        pointcloud.transform(datapoint["X_WC"])

        if scene_voxel_m is not None and float(scene_voxel_m) > 0:
            pointcloud = pointcloud.voxel_down_sample(float(scene_voxel_m))

        pc_list.append(pointcloud)

    if bbox_3d is not None:
        b = np.asarray(bbox_3d, dtype=np.float64)
        if b.shape != (3, 2):
            raise ValueError(
                "bbox_3d must have shape (3, 2): [x_min,x_max], [y_min,y_max], [z_min,z_max] "
                "as rows [[min, max], ...] in world frame."
            )
        lo, hi = b[:, 0], b[:, 1]
        pc_list = [_crop_point_cloud_to_aabb(p, lo, hi) for p in pc_list]

    if robot_points_world is not None:
        r = robot_exclude_radius if robot_exclude_radius is not None else 0.03
        if r > 0:
            tree = _prepare_robot_reference_tree(
                robot_points_world,
                r,
                max_ref_points=robot_ref_max_points,
                ref_voxel_m=robot_ref_voxel_m,
            )
            if tree is not None:
                pc_list = [_remove_points_near_reference_tree(p, tree, r) for p in pc_list]

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
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

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

if __name__ == "__main__":
    serials = ["244622072067", "044322073544"]
    stream = MultiRealSenseStream(serials, "extrinsic_calibration.json")
    pcd_viewer = LivePointCloudViewer()

    for i in range(1000):
        datapoints = stream.get_datapoints()
        fused,_ = get_fused_point_cloud(datapoints)

        # Convert to numpy
        pts = np.asarray(fused.points)
        cols = np.asarray(fused.colors) if fused.has_colors() else None
        pcd_viewer.update(pts, cols)