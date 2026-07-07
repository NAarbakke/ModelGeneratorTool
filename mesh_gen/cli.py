"""mesh_gen CLI: dense reconstruction → optional crop → Poisson mesh → scaled export."""

from __future__ import annotations

import shutil

import click
import numpy as np

from measure3d import project, scaling
from measure3d.cli import workspace_or_die

from . import backends, meshing

FORMATS = ("stl", "obj", "ply", "pointcloud")


@click.group()
def main() -> None:
    """Real-world-scale mesh export from measure3d reconstructions. Units: millimeters."""


@main.command()
@click.argument("photos", required=False, type=click.Path())
@click.option("--format", "fmt", type=click.Choice(FORMATS), default="stl", show_default=True,
              help="stl/obj for CAD & printing, ply keeps color, pointcloud skips meshing.")
@click.option("--depth", type=int, default=9, show_default=True,
              help="Poisson octree depth - higher gives more detail but is slower.")
@click.option("--backend", type=click.Choice(("auto", "cpu", "gpu")), default="auto",
              show_default=True, help="Dense reconstruction backend (auto detects CUDA).")
@click.option("--no-crop", is_flag=True, help="Skip the interactive crop step.")
@click.option("--openmvs-dir", type=click.Path(), envvar=backends.OPENMVS_DIR_ENV,
              help=f"Folder containing the OpenMVS binaries (or set {backends.OPENMVS_DIR_ENV}).")
def export(photos: str | None, fmt: str, depth: int, backend: str, no_crop: bool,
           openmvs_dir: str | None) -> None:
    """Dense-reconstruct the scene and export a mesh at true scale (mm)."""
    ws = workspace_or_die(photos)

    scale = scaling.load_scale(ws.scale_path)
    if scale is None:
        raise click.ClickException(
            "no scale calibration found - an unscaled mesh is useless downstream; "
            "run `measure3d calibrate` first"
        )
    if not ws.has_sparse():
        raise click.ClickException("no sparse reconstruction - run `measure3d reconstruct` first")

    try:
        backend = backends.resolve_backend(backend, openmvs_dir)
        dense_ply = backends.densify(ws, backend, openmvs_dir)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(dense_ply))
    if not len(pcd.points):
        raise click.ClickException(f"dense cloud {dense_ply} is empty - densification failed")
    click.echo(f"Dense cloud: {len(pcd.points):,} points. Applying scale: {scale:.4f} mm/unit.")
    pcd.scale(scale, center=np.zeros(3))

    if not no_crop:
        from measure3d import picking

        click.echo("Crop away background/floor before meshing (close the window to keep all):")
        pcd = picking.interactive_crop(pcd)

    ws.exports_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "pointcloud":
        out_path = ws.exports_dir / f"{ws.name}_pointcloud.ply"
        meshing.write_point_cloud(pcd, out_path)
        stats = f"  points: {len(pcd.points):,}"
    else:
        mesh = meshing.poisson_mesh(pcd, depth=depth)
        out_path = ws.exports_dir / f"{ws.name}.{fmt}"
        meshing.write_mesh(mesh, out_path)
        stats = meshing.mesh_stats(mesh)

    click.echo(f"\nExported to {out_path}  (units: MILLIMETERS - check units on CAD import)")
    click.echo(stats)

    models_dir = project.output_dir(project.OUTPUT_MODELS_DIR)
    if models_dir is not None:
        shutil.copy2(out_path, models_dir / out_path.name)
        click.echo(f"Copied to {models_dir / out_path.name}")


if __name__ == "__main__":
    main()
