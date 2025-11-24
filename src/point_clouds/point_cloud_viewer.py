import os
import sys

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import open3d as o3d
import numpy as np

class LivePointCloudViewer:
    def __init__(self, point_size=2.0):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Live Fused Point Cloud")

        self.pcd = o3d.geometry.PointCloud()

        self.robot = o3d.geometry.PointCloud()

        self.added = False

        opt = self.vis.get_render_option()
        opt.point_size = point_size

    def update(self, pcd_points, end_effector_pos, pcd_colors=None):
        """Update the point cloud in the viewer."""
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.robot.points = o3d.utility.Vector3dVector(end_effector_pos)

        if pcd_colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        if not self.added:
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.robot)
            self.added = True
        else:
            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.robot)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()