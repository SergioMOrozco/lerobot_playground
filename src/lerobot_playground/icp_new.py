#!/usr/bin/env python3
import itertools
import os
import json
import numpy as np
from PIL import Image
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------
# Depth -> point cloud in camera frame
# ------------------------------------------------------------
def depth2pcd(depth, serial, mask=None):
    with open("intrinsic_calibration.json", "r") as f:
        intr = json.load(f)[serial]

    fx, fy = intr["fl_x"], intr["fl_y"]
    cx, cy = intr["cx"], intr["cy"]
    w, h = intr["w"], intr["h"]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    if mask is not None:
        depth = depth.copy()
        depth[mask == 0] = 0.0

    depth_img = o3d.geometry.Image(depth.astype(np.float32))

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img,
        intrinsic,
        depth_scale=1.0,
    )

    return pcd


# ------------------------------------------------------------
# Fixed camera→mesh correspondences (indices into cam_pts and mesh_pts)
# ------------------------------------------------------------
#def compute_fixed_cam_mesh_correspondences(pts_cam, T_init, mesh_kd, mesh_pts, max_dist=1.0):
def compute_fixed_cam_mesh_correspondences(pts_cam, T_init, mesh_kd, mesh_pts, max_dist=0.03):
    """
    pts_cam: (N,3) in camera frame
    T_init: 4x4 world_from_cam initial transform
    Returns:
        cam_idx: indices into pts_cam
        mesh_idx: indices into mesh_pts
    """
    R0, t0 = T_init[:3, :3], T_init[:3, 3]
    pts_world = (R0 @ pts_cam.T).T + t0

    cam_idx = []
    mesh_idx = []
    max_d2 = max_dist * max_dist

    for i, p in enumerate(pts_world):
        k, idx, d2 = mesh_kd.search_knn_vector_3d(p, 1)
        if k == 0:
            continue
        if d2[0] > max_d2:
            continue
        cam_idx.append(i)
        mesh_idx.append(idx[0])

    return np.asarray(cam_idx, dtype=np.int64), np.asarray(mesh_idx, dtype=np.int64)


# ------------------------------------------------------------
# Fixed camera↔camera correspondences (indices into cam_pts)
# ------------------------------------------------------------
def compute_fixed_cam_cam_pairs(pts1_cam, pts2_cam, T1_init, T2_init,
                                max_dist=0.05, max_pairs=2000):
    """
    pts1_cam: (N1,3) camera 1 frame
    pts2_cam: (N2,3) camera 2 frame
    Returns:
        idx1_pairs, idx2_pairs (same length)
        such that pts1_cam[idx1] ↔ pts2_cam[idx2]
    """
    R1, t1 = T1_init[:3, :3], T1_init[:3, 3]
    R2, t2 = T2_init[:3, :3], T2_init[:3, 3]

    pts1_world = (R1 @ pts1_cam.T).T + t1
    pts2_world = (R2 @ pts2_cam.T).T + t2

    pc2_world = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts2_world))
    kd2 = o3d.geometry.KDTreeFlann(pc2_world)

    idx1_pairs = []
    idx2_pairs = []

    max_d2 = max_dist * max_dist

    n1 = pts1_world.shape[0]
    if n1 == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    N = min(max_pairs, n1)
    sample_idx = np.random.choice(n1, size=N, replace=False)

    for i in sample_idx:
        p = pts1_world[i]
        k, idx, d2 = kd2.search_knn_vector_3d(p, 1)
        if k == 0:
            continue
        if d2[0] > max_d2:
            continue

        j = idx[0]
        idx1_pairs.append(i)
        idx2_pairs.append(j)

    return np.asarray(idx1_pairs, dtype=np.int64), np.asarray(idx2_pairs, dtype=np.int64)


def discover_calibration_serials(calib_dir):
    """Serials with both depth.npz and mask.png under calib_dir/<serial>/."""
    serials = []
    for name in sorted(os.listdir(calib_dir)):
        d = os.path.join(calib_dir, name)
        if (
            os.path.isdir(d)
            and os.path.exists(os.path.join(d, "depth.npz"))
            and os.path.exists(os.path.join(d, "mask.png"))
        ):
            serials.append(name)
    return serials


# ------------------------------------------------------------
# Param <-> transform helpers
# ------------------------------------------------------------
def params_to_transforms(p, n_cams):
    """
    p: (6 * n_cams,) = [r_0(3), t_0(3), r_1(3), t_1(3), ...]
    """
    p = p.reshape(n_cams, 6)
    transforms = []
    for r, t in zip(p[:, 0:3], p[:, 3:6]):
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(r).as_matrix()
        T[:3, 3] = t
        transforms.append(T)
    return transforms


def matrix_to_rotvec(T):
    return R.from_matrix(T[:3, :3]).as_rotvec()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    calib_dir = "calibration_files"

    # -------- Load extrinsics --------
    with open("extrinsic_calibration.json", "r") as f:
        extr = json.load(f)

    # -------- Discover cameras with calibration data on disk --------
    available = discover_calibration_serials(calib_dir)
    serials = [s for s in available if s in extr]
    missing = sorted(set(available) - set(extr))
    if missing:
        print(f"Warning: skipping {missing} (no entry in extrinsic_calibration.json)")
    if not serials:
        raise RuntimeError(f"No calibration_files serials with extrinsics found in {calib_dir}")

    print("Calibrating cameras:", serials)

    # -------- Load robot mesh --------
    mesh_npz = os.path.join(calib_dir, "robot_pcd.npz")
    with np.load(mesh_npz) as data:
        mesh_pts = data["pcd"]

    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_pts))
    mesh = mesh.voxel_down_sample(0.005)
    mesh.estimate_normals()

    mesh_points = np.asarray(mesh.points)
    mesh_normals = np.asarray(mesh.normals)
    mesh_kd = o3d.geometry.KDTreeFlann(mesh)

    # -------- Load camera point clouds (ONCE, fixed) --------
    cam_pts = {}

    for serial in serials:
        dfile = os.path.join(calib_dir, serial, "depth.npz")
        mfile = os.path.join(calib_dir, serial, "mask.png")

        with np.load(dfile) as data:
            depth = data["depth"].astype(np.float32) / 1000.0  # mm->m

        mask = np.array(Image.open(mfile))[..., 3]

        pcd = depth2pcd(depth, serial, mask=mask)
        pcd = pcd.voxel_down_sample(0.004)

        pts = np.asarray(pcd.points)
        cam_pts[serial] = pts.copy()

        print(f"{serial}: using {pts.shape[0]} fixed points")

    # -------- Initial parameters --------
    T_init = {}
    params0 = np.zeros(6 * len(serials), dtype=np.float64)
    for i, serial in enumerate(serials):
        T_init[serial] = np.asarray(extr[serial]["X_WC"])
        params0[6 * i : 6 * i + 3] = matrix_to_rotvec(T_init[serial])
        params0[6 * i + 3 : 6 * i + 6] = T_init[serial][:3, 3]

    # -------- Fixed cam->mesh correspondences --------
    cam_mesh_corr = {
        serial: compute_fixed_cam_mesh_correspondences(
            cam_pts[serial], T_init[serial], mesh_kd, mesh_points
        )
        for serial in serials
    }
    for serial, (cam_idx, mesh_idx) in cam_mesh_corr.items():
        print(f"{serial}: {len(cam_idx)} cam->mesh correspondences")

    # -------- Fixed cam<->cam correspondences (index pairs into cam_pts) --------
    pairwise_corr = {}
    for s_i, s_j in itertools.combinations(serials, 2):
        idx_i, idx_j = compute_fixed_cam_cam_pairs(
            cam_pts[s_i], cam_pts[s_j], T_init[s_i], T_init[s_j]
        )
        pairwise_corr[(s_i, s_j)] = (idx_i, idx_j)
        print(f"Camera-camera pairs {s_i}<->{s_j}: {idx_i.shape[0]}")

    # Weights
    w_icp = 1.0
    w_cc = 0.1  # camera-camera consistency weight

    # -------- Residual function --------
    def residuals(p):
        transforms = params_to_transforms(p, len(serials))
        world_pts = {}
        for serial, T in zip(serials, transforms):
            Rm, t = T[:3, :3], T[:3, 3]
            world_pts[serial] = (Rm @ cam_pts[serial].T).T + t

        parts = []

        # --- camera->mesh ICP residuals (point-to-plane) ---
        for serial in serials:
            cam_idx, mesh_idx = cam_mesh_corr[serial]
            q = mesh_points[mesh_idx]
            n = mesh_normals[mesh_idx]
            parts.append(w_icp * np.sum(n * (world_pts[serial][cam_idx] - q), axis=1))

        # --- camera<->camera consistency (3D diffs at index pairs) ---
        for (s_i, s_j), (idx_i, idx_j) in pairwise_corr.items():
            if idx_i.size == 0:
                continue
            parts.append(w_cc * (world_pts[s_i][idx_i] - world_pts[s_j][idx_j]).reshape(-1))

        return np.concatenate(parts)

    # -------- Optimize --------
    print("Running joint ICP optimization...")
    result = least_squares(
        residuals,
        params0,
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=200,
        verbose=2,
    )
    print("Optimization finished.")
    print("Initial residual norm:", np.linalg.norm(residuals(params0)))
    print("Final   residual norm:", np.linalg.norm(residuals(result.x)))

    p_opt = result.x
    T_opt = params_to_transforms(p_opt, len(serials))

    # -------- Save updated extrinsics --------
    # Merge into the existing extrinsics rather than overwrite, so a camera
    # not present in calibration_files this run (e.g. temporarily unplugged)
    # keeps its previous entry instead of being dropped from the file.
    for serial, T in zip(serials, T_opt):
        extr[serial] = {"X_WC": T.tolist()}

    with open("extrinsic_calibration.json", "w") as f:
        json.dump(extr, f, indent=4)

    print("Saved refined extrinsics to extrinsic_calibration.json")

    # -------- Visualize merged point clouds (using same cam_pts) --------
    merged = o3d.geometry.PointCloud()

    for serial, T in zip(serials, T_opt):
        Rw, tw = T[:3, :3], T[:3, 3]
        pts_cam = cam_pts[serial]
        pts_world = (Rw @ pts_cam.T).T + tw
        pcd_world = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
        merged += pcd_world

    o3d.visualization.draw_geometries([merged, mesh])


if __name__ == "__main__":
    main()
