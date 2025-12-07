#!/usr/bin/env python

import os
import json
import numpy as np
from PIL import Image

import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from point_clouds.point_cloud_viewer import LivePointCloudViewer


# ------------------------------
# Depth → PointCloud (your version, lightly cleaned)
# ------------------------------
def depth2pcd(depth, serial, color=None, T_wc=None, mask=None):
    with open("intrinsic_calibration.json", "r") as f:
        intrinsics_cfg = json.load(f)

    if mask is not None:
        depth = depth.copy()
        depth[mask == 0] = 0.0

    fl_x = intrinsics_cfg[serial]['fl_x']
    fl_y = intrinsics_cfg[serial]['fl_y']
    cx = intrinsics_cfg[serial]['cx']
    cy = intrinsics_cfg[serial]['cy']
    w = intrinsics_cfg[serial]['w']
    h = intrinsics_cfg[serial]['h']

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

    depth = np.ascontiguousarray(depth.astype(np.float32))
    depth_image = o3d.geometry.Image(depth)

    if color is not None:
        if color.dtype == np.float32:
            img_uint8 = np.array(color * 255, dtype=np.uint8)
        else:
            img_uint8 = np.array(color)
        color_image = o3d.geometry.Image(img_uint8)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic,
        )
    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic,
            depth_scale=1.0,
        )

    if T_wc is not None:
        pcd.transform(T_wc)

    return pcd


# ------------------------------
# Rotation parameterization helpers
# ------------------------------
def params_to_poses(params):
    """
    params: [r1(3), t1(3), r2(3), t2(3)]
    Returns T1, T2 (4x4 each)
    """
    r1 = params[0:3]
    t1 = params[3:6]
    r2 = params[6:9]
    t2 = params[9:12]

    T1 = np.eye(4)
    T1[:3, :3] = R.from_rotvec(r1).as_matrix()
    T1[:3, 3] = t1

    T2 = np.eye(4)
    T2[:3, :3] = R.from_rotvec(r2).as_matrix()
    T2[:3, 3] = t2

    return T1, T2


def matrix_to_rotvec(T):
    """
    T: 4x4 homogeneous transform
    returns rotation vector (3,)
    """
    return R.from_matrix(T[:3, :3]).as_rotvec()


# ------------------------------
# Main
# ------------------------------
def main():
    calibration_dir = "calibration_files"

    # ---------- Load extrinsics ----------
    with open("extrinsic_calibration.json", "r") as f:
        extrinsics = json.load(f)

    serials = list(extrinsics.keys())
    serials.sort()  # deterministic order

    if len(serials) != 2:
        raise ValueError(f"This script expects exactly 2 cameras, found {len(serials)}: {serials}")

    serial_a, serial_b = serials[0], serials[1]
    print(f"Using cameras: {serial_a}, {serial_b}")

    # ---------- Load robot mesh point cloud ----------
    robot_pcd_path = os.path.join(calibration_dir, "robot_pcd.npz")
    if not os.path.exists(robot_pcd_path):
        raise FileNotFoundError(f"Could not find {robot_pcd_path}")

    with np.load(robot_pcd_path) as data:
        robot_pcd = data["pcd"]

    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(robot_pcd)
    mesh_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # green

    # Downsample mesh for speed & stability
    mesh_voxel = 0.005
    mesh_pcd = mesh_pcd.voxel_down_sample(mesh_voxel)
    mesh_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=mesh_voxel * 2.0, max_nn=30)
    )

    mesh_points = np.asarray(mesh_pcd.points)
    mesh_normals = np.asarray(mesh_pcd.normals)

    # KD-tree on mesh for initial correspondences
    mesh_kd = o3d.geometry.KDTreeFlann(mesh_pcd)

    # ---------- Load camera point clouds ----------
    camera_pcds = {}
    camera_points = {}

    # limit points per camera for optimization
    max_cam_points = 8000

    for serial in serials:
        serial_dir = os.path.join(calibration_dir, serial)

        depth_path = os.path.join(serial_dir, "depth.npz")
        mask_path = os.path.join(serial_dir, "mask.png")

        if not os.path.exists(depth_path):
            raise FileNotFoundError(depth_path)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        with np.load(depth_path) as data:
            depth = data["depth"].astype(np.float32) / 1000.0  # assume mm → meters
            #depth = data["depth"].astype(np.float32)  # assume mm → meters

        mask = np.array(Image.open(mask_path))[..., 3]

        print(f"Building camera-frame point cloud for {serial}")
        pcd_cam = depth2pcd(depth, serial, color=None, T_wc=None, mask=mask)
        pcd_cam = pcd_cam.remove_radius_outlier(nb_points=25, radius=0.01)[0]

        # Downsample for speed
        cam_voxel = 0.004
        pcd_cam = pcd_cam.voxel_down_sample(cam_voxel)

        # Randomly reduce to max_cam_points
        pts = np.asarray(pcd_cam.points)
        if pts.shape[0] > max_cam_points:
            idx = np.random.choice(pts.shape[0], size=max_cam_points, replace=False)
            pts = pts[idx]
            pcd_cam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        camera_pcds[serial] = pcd_cam
        camera_points[serial] = np.asarray(pcd_cam.points)

        print(f"{serial}: {camera_points[serial].shape[0]} points after filtering")

    # ---------- Initial transforms and parameter vector ----------
    T_init = {}
    for serial in serials:
        T = np.asarray(extrinsics[serial]["X_WC"], dtype=np.float64)
        T_init[serial] = T

    # Params: [r1, t1, r2, t2]
    params_init = []
    for serial in [serial_a, serial_b]:
        T = T_init[serial]
        r = matrix_to_rotvec(T)
        t = T[:3, 3]
        params_init.extend(r.tolist())
        params_init.extend(t.tolist())
    params_init = np.array(params_init, dtype=np.float64)

    # ---------- Precompute correspondences: camera → mesh ----------
    print("Precomputing camera → mesh correspondences...")
    max_mesh_dist = 0.03  # 3 cm cutoff

    cam_mesh_src = {}   # serial -> (N_i, 3) cam points used
    cam_mesh_idx = {}   # serial -> (N_i,) mesh indices

    for serial in serials:
        pts_cam = camera_points[serial]

        T0 = T_init[serial]
        R0 = T0[:3, :3]
        t0 = T0[:3, 3]

        pts_world = (R0 @ pts_cam.T).T + t0

        selected_cam_pts = []
        selected_mesh_idx = []

        for p in pts_world:
            k, idx, dists = mesh_kd.search_knn_vector_3d(p, 1)
            if k == 0:
                continue
            dist2 = dists[0]
            if dist2 > max_mesh_dist ** 2:
                continue
            j = idx[0]
            selected_cam_pts.append(p)   # store world-space point at init (only for diagnostics)
            selected_mesh_idx.append(j)

        # But we want camera-frame points for optimization, not the world ones:
        # So recompute using indices
        selected_cam_pts_cam = []
        for p_cam, p_w_init, j in zip(pts_cam, selected_cam_pts, selected_mesh_idx):
            # p_cam is already cam-frame, we can just use it directly
            selected_cam_pts_cam.append(p_cam)

        cam_mesh_src[serial] = np.asarray(selected_cam_pts_cam, dtype=np.float64)
        cam_mesh_idx[serial] = np.asarray(selected_mesh_idx, dtype=np.int64)

        print(
            f"{serial}: {cam_mesh_src[serial].shape[0]} cam→mesh correspondences"
        )

    # ---------- Precompute camera ↔ camera correspondences ----------
    print("Precomputing camera ↔ camera correspondences...")
    pts_a_cam = camera_points[serial_a]
    pts_b_cam = camera_points[serial_b]

    T_a0 = T_init[serial_a]
    T_b0 = T_init[serial_b]
    Ra0, ta0 = T_a0[:3, :3], T_a0[:3, 3]
    Rb0, tb0 = T_b0[:3, :3], T_b0[:3, 3]

    pts_a_world_init = (Ra0 @ pts_a_cam.T).T + ta0
    pts_b_world_init = (Rb0 @ pts_b_cam.T).T + tb0

    pc_b_world = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_b_world_init))
    kd_b_world = o3d.geometry.KDTreeFlann(pc_b_world)

    max_pairs = 2000
    n_a = pts_a_world_init.shape[0]
    n_sample = min(max_pairs, n_a)

    if n_sample == 0:
        raise RuntimeError("No points for camera-camera correspondences")

    idx_a_sample = np.random.choice(n_a, size=n_sample, replace=False)

    pair_pts_a_cam = []
    pair_pts_b_cam = []

    max_cc_dist = 0.03  # 3 cm

    for i in idx_a_sample:
        p1 = pts_a_world_init[i]
        k, idx, dists = kd_b_world.search_knn_vector_3d(p1, 1)
        if k == 0:
            continue
        dist2 = dists[0]
        if dist2 > max_cc_dist ** 2:
            continue
        j = idx[0]
        pair_pts_a_cam.append(pts_a_cam[i])
        pair_pts_b_cam.append(pts_b_cam[j])

    cam_pair_pts_a = np.asarray(pair_pts_a_cam, dtype=np.float64)
    cam_pair_pts_b = np.asarray(pair_pts_b_cam, dtype=np.float64)

    print(
        f"Camera-camera: {cam_pair_pts_a.shape[0]} correspondences between {serial_a} and {serial_b}"
    )

    if cam_pair_pts_a.shape[0] == 0:
        raise RuntimeError("No valid camera-camera correspondences; check setup/overlap")

    # ---------- Build cost function ----------
    w_icp = 1.0
    w_cc = 0.5
    w_reg = 0.01

    def cost_function(params):
        T1, T2 = params_to_poses(params)
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R2, t2 = T2[:3, :3], T2[:3, 3]

        residuals = []

        # --- Camera A → mesh (point-to-plane)
        P_cam = cam_mesh_src[serial_a]  # (Na, 3)
        idx_mesh = cam_mesh_idx[serial_a]
        if P_cam.shape[0] > 0:
            Pw = (R1 @ P_cam.T).T + t1  # (Na, 3)
            Q = mesh_points[idx_mesh]
            N = mesh_normals[idx_mesh]
            r_a = np.sum(N * (Pw - Q), axis=1)  # (Na,)
            residuals.append(w_icp * r_a)

        # --- Camera B → mesh (point-to-plane)
        P_cam = cam_mesh_src[serial_b]  # (Nb, 3)
        idx_mesh = cam_mesh_idx[serial_b]
        if P_cam.shape[0] > 0:
            Pw = (R2 @ P_cam.T).T + t2  # (Nb, 3)
            Q = mesh_points[idx_mesh]
            N = mesh_normals[idx_mesh]
            r_b = np.sum(N * (Pw - Q), axis=1)  # (Nb,)
            residuals.append(w_icp * r_b)

        # --- Camera-camera consistency (3D point difference)
        Pa = cam_pair_pts_a  # (K, 3) in cam A frame
        Pb = cam_pair_pts_b  # (K, 3) in cam B frame
        if Pa.shape[0] > 0:
            Pa_w = (R1 @ Pa.T).T + t1
            Pb_w = (R2 @ Pb.T).T + t2
            diff = Pa_w - Pb_w  # (K, 3)
            residuals.append(w_cc * diff.reshape(-1))  # flatten

        # --- Regularization toward initial params (tiny pull-back)
        reg = w_reg * (params - params_init)
        residuals.append(reg)

        return np.concatenate(residuals)

    # ---------- Run optimization ----------
    print("Running SciPy joint bundle adjustment (2 cameras + mesh)...")

    result = least_squares(
        cost_function,
        params_init,
        method="lm",   # Levenberg–Marquardt
        max_nfev=200,
        verbose=2,
    )

    print("Optimization done.")
    print("Initial cost norm:", np.linalg.norm(cost_function(params_init)))
    print("Final   cost norm:", np.linalg.norm(cost_function(result.x)))

    params_opt = result.x
    T1_opt, T2_opt = params_to_poses(params_opt)

    print("T1_opt (world ←", serial_a, "):\n", T1_opt)
    print("T2_opt (world ←", serial_b, "):\n", T2_opt)

    # ---------- Save refined extrinsics ----------
    new_extrinsics = {
        serial_a: {"X_WC": T1_opt.tolist()},
        serial_b: {"X_WC": T2_opt.tolist()},
    }

    with open("extrinsic_calibration.json", "w") as f:
        json.dump(new_extrinsics, f, indent=4)

    print("Saved refined extrinsics to extrinsic_calibration.json")

    # ---------- Quick visual sanity check ----------
    # Rebuild colored point clouds in world frame using refined extrinsics
    print("Visualizing merged refined clouds vs mesh...")
    merged_pc = o3d.geometry.PointCloud()

    for serial, T_opt in zip([serial_a, serial_b], [T1_opt, T2_opt]):
        serial_dir = os.path.join(calibration_dir, serial)

        with np.load(os.path.join(serial_dir, "depth.npz")) as data:
            #depth = data["depth"].astype(np.float32) / 1000.0
            depth = data["depth"].astype(np.float32)

        color = np.array(Image.open(os.path.join(serial_dir, "color.png")))
        mask = np.array(Image.open(os.path.join(serial_dir, "mask.png")))[..., 3]

        pcd_world = depth2pcd(depth, serial, color=color, T_wc=T_opt, mask=mask)
        merged_pc += pcd_world

    o3d.visualization.draw_geometries([merged_pc, mesh_pcd])


if __name__ == "__main__":
    serials = ["244622072067", "044322073544"]
    stream = MultiRealSenseStream(serials, "extrinsic_calibration.json")
    pcd_viewer = LivePointCloudViewer()

    for i in range(100):
        datapoints = stream.get_datapoints()
        fused,_ = get_fused_point_cloud(datapoints)

        # Convert to numpy
        pts = np.asarray(fused.points)
        cols = np.asarray(fused.colors) if fused.has_colors() else None
        pcd_viewer.update(pts, cols)

    stream.stop()
