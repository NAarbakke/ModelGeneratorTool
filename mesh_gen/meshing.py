"""Dense point cloud → triangle mesh: normals, Poisson reconstruction, trim, export.

Open3D is imported lazily so `--help`, error paths, and the smoke test stay fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TRIM_QUANTILE = 0.01  # Poisson vertices below this density quantile are blob artifacts


def low_density_mask(densities: np.ndarray, quantile: float = TRIM_QUANTILE) -> np.ndarray:
    """Boolean mask of Poisson vertices to remove (low support in the input cloud)."""
    d = np.asarray(densities, dtype=float)
    return d < np.quantile(d, quantile)


def ensure_normals(pcd) -> None:
    """Dense clouds from OpenMVS/COLMAP fusion already carry oriented normals;
    estimate them only as a fallback."""
    import open3d as o3d

    if pcd.has_normals():
        return
    print("Point cloud has no normals - estimating (this can take a while)...")
    nn = np.mean(pcd.compute_nearest_neighbor_distance())
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3 * nn, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)


def poisson_mesh(pcd, depth: int = 9):
    """Poisson surface reconstruction with low-density trimming. Cloud must be scaled to mm."""
    import open3d as o3d

    ensure_normals(pcd)
    print(f"Running Poisson surface reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    mesh.remove_vertices_by_mask(low_density_mask(np.asarray(densities)))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def write_mesh(mesh, path: Path) -> None:
    import open3d as o3d

    if path.suffix.lower() == ".stl":
        mesh.compute_triangle_normals()  # STL export fails without them
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise RuntimeError(f"failed to write mesh to {path}")


def write_point_cloud(pcd, path: Path) -> None:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise RuntimeError(f"failed to write point cloud to {path}")


def mesh_stats(mesh) -> str:
    """Human-readable stats block; all lengths in mm."""
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    return (
        f"  vertices:   {len(mesh.vertices):,}\n"
        f"  triangles:  {len(mesh.triangles):,}\n"
        f"  watertight: {'yes' if mesh.is_watertight() else 'no'}\n"
        f"  bbox (mm):  {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f}"
    )
