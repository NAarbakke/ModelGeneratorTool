"""Interactive Open3D point picking and cropping (legacy editing visualizer).

Open3D is imported lazily: it takes seconds to import and needs a display,
so CLI help/error paths must not touch it.
"""

from __future__ import annotations

import numpy as np

PICK_HELP = """\
Viewer controls:
  Shift + left click    pick a point (Shift + right click to undo)
  Q / Esc               close the window when done"""

CROP_HELP = """\
Viewer controls:
  K                     lock the view and enter selection mode
  mouse drag            rectangle-select (Ctrl + click for polygon)
  C                     crop to the selection
  F                     back to free view
  Q / Esc               close the window when done"""


def make_cloud(points: np.ndarray, colors: np.ndarray | None = None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return pcd


def pick_points(pcd, window_name: str = "pick points") -> list[int]:
    """Open the picking viewer; blocks until closed. Returns picked point indices."""
    import open3d as o3d

    print(PICK_HELP)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=1280, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return list(vis.get_picked_points())


def interactive_crop(pcd):
    """Open the crop viewer; blocks until closed. Returns the cropped cloud
    (the original if nothing was cropped)."""
    import open3d as o3d

    print(CROP_HELP)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="crop (K, drag, C, then Q)", width=1280, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    cropped = vis.get_cropped_geometry()
    if cropped is None or not len(cropped.points) or len(cropped.points) >= len(pcd.points):
        print("No crop applied - keeping the full cloud.")
        return pcd
    print(f"Cropped {len(pcd.points):,} -> {len(cropped.points):,} points.")
    return cropped
