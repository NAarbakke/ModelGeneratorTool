"""measure3d CLI: reconstruct, calibrate, measure. All real-world lengths are in mm."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import click
import numpy as np

from . import project, scaling


@click.group()
def main() -> None:
    """Real-world dimensions of an object from photos. Units: millimeters."""


def workspace_or_die(photos: str | None) -> project.Workspace:
    try:
        return project.Workspace(project.resolve_photos_dir(photos))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))


def sparse_points_or_die(ws: project.Workspace) -> tuple[np.ndarray, np.ndarray]:
    from . import reconstruction

    try:
        return reconstruction.load_sparse_points(ws)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))


@main.command()
@click.argument("photos", required=False, type=click.Path())
def reconstruct(photos: str | None) -> None:
    """Run sparse structure-from-motion on a photo folder (default: ./input_photos)."""
    from . import reconstruction

    ws = workspace_or_die(photos)
    try:
        s = reconstruction.run_sfm(ws)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"\nRegistered {s.registered_images}/{s.total_images} images, "
        f"{s.num_points:,} sparse points, "
        f"mean reprojection error {s.mean_reprojection_error:.2f} px"
    )
    click.echo(f"Reconstruction saved to {ws.sparse_model_dir}")
    if s.registered_images < max(3, s.total_images // 2):
        raise click.ClickException(
            f"only {s.registered_images} of {s.total_images} images registered - add more "
            "overlapping photos or improve texture/lighting (the partial model was kept)"
        )
    click.echo("Next: measure3d calibrate --known-distance <mm>")


@main.command()
@click.argument("photos", required=False, type=click.Path())
@click.option(
    "--known-distance",
    type=float,
    default=None,
    help="Real distance in mm between the two points you pick (prompted if omitted).",
)
def calibrate(photos: str | None, known_distance: float | None) -> None:
    """Set the real-world scale by picking two points of known distance."""
    from . import picking

    ws = workspace_or_die(photos)
    points, colors = sparse_points_or_die(ws)
    pcd = picking.make_cloud(points, colors)

    click.echo("Pick exactly 2 points with a known real-world distance (e.g. ends of a ruler).")
    while True:
        picked = picking.pick_points(pcd, "measure3d calibrate - pick 2 points")
        if len(picked) == 2:
            break
        click.echo(f"Picked {len(picked)} point(s); exactly 2 are needed - reopening viewer.")

    picked_units = scaling.distance(points[picked[0]], points[picked[1]])
    if known_distance is None:
        known_distance = click.prompt("Real-world distance between the points (mm)", type=float)
    try:
        scale = scaling.compute_scale(known_distance, picked_units)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    scaling.save_scale(ws.scale_path, scale, known_distance, picked_units)
    click.echo(f"Scale set: 1 reconstruction unit = {scale:.4f} mm")
    click.echo(f"Saved to {ws.scale_path} - re-measuring won't need re-calibration.")


@main.command()
@click.argument("photos", required=False, type=click.Path())
@click.option("--crop", is_flag=True, help="Interactively crop away background/floor first.")
def measure(photos: str | None, crop: bool) -> None:
    """Interactive measuring: pick point pairs, get real distances in mm."""
    from . import picking

    ws = workspace_or_die(photos)
    scale = scaling.load_scale(ws.scale_path)
    if scale is None:
        raise click.ClickException("no calibration found - run `measure3d calibrate` first")
    points, colors = sparse_points_or_die(ws)
    pcd = picking.make_cloud(points, colors)

    if crop:
        pcd = picking.interactive_crop(pcd)
        points = np.asarray(pcd.points)

    distances: list[float] = []
    while True:
        click.echo("\nPick an even number of points; each consecutive pair is one measurement.")
        picked = picking.pick_points(pcd, "measure3d - pick point pairs")
        if len(picked) % 2:
            click.echo("(odd number of picks - the last point is ignored)")
        pair_dists = scaling.pair_distances_mm(points, picked, scale)
        for i, dist in enumerate(pair_dists, start=len(distances) + 1):
            click.echo(f"  measurement {i}: {dist:.2f} mm")
        distances.extend(pair_dists)
        if not click.confirm("Measure more pairs?", default=False):
            break

    mn, mx = scaling.scaled_aabb_mm(points, scale)
    extent = mx - mn
    click.echo(f"\nBounding box (mm): {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f}")
    if not crop:
        click.echo("  warning: box includes background points - re-run with --crop to isolate the object")

    _write_report(ws, distances, extent, cropped=crop)


def _write_report(ws: project.Workspace, distances: list[float], extent: np.ndarray, cropped: bool) -> None:
    out_dir = project.output_dir(project.OUTPUT_DIMS_DIR)
    if out_dir is None:
        return
    report = {
        "photos": str(ws.photos_dir.resolve()),
        "units": "mm",
        "measured_distances_mm": [round(d, 3) for d in distances],
        "bounding_box_mm": [round(float(v), 2) for v in extent],
        "cropped": cropped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = out_dir / f"{ws.name}-measurements.json"
    path.write_text(json.dumps(report, indent=2))
    click.echo(f"Report written to {path}")


if __name__ == "__main__":
    main()
