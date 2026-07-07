"""measure2d CLI: real-world in-plane measurements from a single photo."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np

from measure3d import project

from . import geometry


@click.command()
@click.argument("photo", type=click.Path(exists=True, dir_okay=False))
@click.option("--ref", default="a4", show_default=True,
              help="Reference rectangle visible in the photo: a4, letter, credit-card, "
                   "or WIDTHxHEIGHT in mm (e.g. 120x80).")
@click.option("--ref-length", type=float, default=None,
              help="Line mode: real distance in mm between TWO clicked points (e.g. ruler "
                   "marks). Only accurate when shooting square-on to the surface.")
def main(photo: str, ref: str, ref_length: float | None) -> None:
    """Measure real distances (mm) in the plane of a reference object in PHOTO.

    A viewer opens: first click the reference (4 corners, or 2 ends with
    --ref-length), then click point pairs to measure. Distances are only valid
    for points lying in the same plane as the reference.
    """
    from . import gui  # deferred: imports OpenCV, which wants a display

    try:
        image = gui.load_image(photo)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    try:
        if ref_length is not None:
            to_metric = _line_reference(image, ref_length)
            click.echo("Line mode: accurate only when the photo is square-on to the surface.")
        else:
            to_metric = _rect_reference(image, ref)
        click.echo("Now click point pairs to measure (u = undo, Enter or q = done).")
        distances, annotated = gui.measure_loop(image, to_metric)
    except gui.Canceled:
        raise click.ClickException("canceled")

    if not distances:
        click.echo("No measurements taken.")
        return
    click.echo(f"\n{len(distances)} measurement(s) taken. Units: millimeters.")
    reference = f"line:{ref_length}mm" if ref_length is not None else ref
    _write_report(Path(photo), reference, distances, annotated)


def _rect_reference(image, ref: str):
    """Click the reference rectangle's corners; returns a pixel->mm plane mapping."""
    from . import gui

    width, height = _ref_dims(ref)
    click.echo(
        f"Click the 4 corners of the {ref} reference ({width:g} x {height:g} mm) "
        "in order: top-left, top-right, bottom-right, bottom-left (as seen in the photo)."
    )
    labels = [f"{name} corner of {ref}" for name in ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")]
    corners = gui.collect_points(image, labels)
    try:
        H = geometry.solve_homography(corners, geometry.rect_corners(width, height))
    except np.linalg.LinAlgError:
        raise click.ClickException(
            "the clicked corners are degenerate (collinear or repeated) - run again and "
            "click the 4 distinct corners of the reference"
        )
    return lambda points: geometry.apply_homography(H, points)


def _line_reference(image, ref_length: float):
    """Click two points a known distance apart; returns a uniform pixel->mm mapping."""
    from . import gui

    if ref_length <= 0:
        raise click.ClickException("--ref-length must be positive (mm)")
    click.echo(f"Click the two ends of your {ref_length:g} mm reference.")
    ends = gui.collect_points(image, ["reference end 1", "reference end 2"])
    pixel_length = float(np.linalg.norm(ends[1] - ends[0]))
    if pixel_length < 1.0:
        raise click.ClickException("reference points coincide - click two distinct points")
    mm_per_px = ref_length / pixel_length
    return lambda points: np.asarray(points, dtype=float) * mm_per_px


def _ref_dims(ref: str) -> tuple[float, float]:
    key = ref.lower()
    if key in geometry.REFERENCE_PRESETS:
        return geometry.REFERENCE_PRESETS[key]
    match = re.fullmatch(r"(\d+\.?\d*)x(\d+\.?\d*)", key)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise click.ClickException(
        f"unknown reference '{ref}' - use a4, letter, credit-card, or WIDTHxHEIGHT in mm"
    )


def _write_report(photo: Path, reference: str, distances: list[float], annotated) -> None:
    import cv2

    out_dir = project.output_dir(project.OUTPUT_DIMS_DIR) or photo.parent
    image_path = out_dir / f"{photo.stem}-annotated.png"
    cv2.imwrite(str(image_path), annotated)
    report = {
        "photo": str(photo.resolve()),
        "reference": reference,
        "units": "mm",
        "measured_distances_mm": [round(d, 2) for d in distances],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report_path = out_dir / f"{photo.stem}-2d-measurements.json"
    report_path.write_text(json.dumps(report, indent=2))
    click.echo(f"Annotated image: {image_path}")
    click.echo(f"Report: {report_path}")


if __name__ == "__main__":
    main()
