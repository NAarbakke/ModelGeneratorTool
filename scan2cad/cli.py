"""scan2cad CLI: video -> frames -> SfM -> ChArUco scale -> SAM 2 mask -> object cloud
-> primitives -> CAD. All lengths in mm.

Typical run:
    scan2cad board                                   # print it, check the square size
    scan2cad frames input/videos/part.mp4            # -> output/scans/part
    scan2cad reconstruct output/scans/part
    scan2cad align output/scans/part
    scan2cad mask output/scans/part                  # click the part
    scan2cad cloud output/scans/part
    scan2cad segment output/scans/part
    scan2cad agent output/scans/part --runner claude
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from core.cliutil import board_options, make_spec

from . import workspace as wsmod

# Imported by the pipeline beyond the shared core; shipped as the `scan2cad` extra.
EXTRA_MODULES = ("scipy", "skimage", "trimesh", "cadquery", "mcp")


def scan_or_die(scan: str) -> wsmod.ScanWorkspace:
    try:
        return wsmod.resolve_scan(scan)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))


def echo_json(obj) -> None:
    click.echo(json.dumps(obj, indent=2, default=str))


@click.group()
def main() -> None:
    """Scan-to-CAD from a video of a part on a ChArUco board. Units: millimeters."""
    import importlib.util

    missing = [m for m in EXTRA_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise click.ClickException(
            f"missing {', '.join(missing)} - install the extra: pip install -e \".[scan2cad]\""
        )


# --- capture ------------------------------------------------------------------------

@main.command()
@click.option("--out", type=click.Path(), default="output/board.png", show_default=True)
@click.option("--dpi", type=int, default=300, show_default=True)
@board_options
def board(out, dpi, **opts) -> None:
    """Write a printable ChArUco board. Print at 100% and measure a square before use."""
    from core import board as boardlib

    from .evaluation import render

    spec = make_spec(**opts)
    img = boardlib.board_image(spec, dpi / 25.4, margin_mm=8)
    path = render.save_png(img[..., None].repeat(3, axis=2), Path(out))
    Path(out).with_suffix(".json").write_text(json.dumps(spec.to_dict(), indent=2))
    w, h = spec.size_mm
    click.echo(f"Board written to {path} ({w:.0f} x {h:.0f} mm + margin at {dpi} dpi).")
    click.echo("Print at 100% (no 'fit to page'), glue it flat, measure one square and pass "
               "--square-mm to `scan2cad align` if it differs from "
               f"{spec.square_mm} mm.")


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(), default=None, help="Scan folder [output/scans/<video>].")
@click.option("--count", type=int, default=100, show_default=True, help="Frames to keep.")
@click.option("--start", "start_s", type=float, default=None, help="Range start (s).")
@click.option("--end", "end_s", type=float, default=None, help="Range end (s).")
def frames(video, out, count, start_s, end_s) -> None:
    """Extract sharp, evenly spread frames from a video into a new scan folder."""
    from .capture import frames as fr

    video = Path(video)
    ws = wsmod.ScanWorkspace(Path(out) if out else wsmod.default_scan_dir(video))
    try:
        meta = fr.extract(video, ws.frames_dir, count, start_s, end_s)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    ws.update_meta(**meta)
    click.echo(f"Kept {len(meta['source_frames'])} frames from {meta['range_s'][0]:.1f}-"
               f"{meta['range_s'][1]:.1f} s ({meta['dropped_blurry']} dropped as blurry) "
               f"-> {ws.frames_dir}")
    click.echo(f"Next: scan2cad reconstruct {ws.root}")


@main.command()
@click.argument("part")
@click.argument("out", type=click.Path())
@click.option("--frames", "n", type=int, default=120, show_default=True)
def synth(part, out, n) -> None:
    """Render a synthetic capture of a known PART (for testing): video + ground truth."""
    from .synthetic import capture as sy
    from .synthetic.parts import PARTS

    if part not in PARTS:
        raise click.ClickException(f"unknown part; choose from {', '.join(PARTS)}")
    gt = sy.render_capture(part, Path(out), frames=n)
    x, y = gt["click_frame0_px"]
    click.echo(f"Wrote {Path(out) / 'capture.mp4'} and ground truth in {Path(out) / 'gt'}")
    click.echo(f"The part's center in video frame 0 is at pixel ({x:.0f}, {y:.0f}).")


# --- reconstruction -----------------------------------------------------------------

@main.command()
@click.argument("scan", type=click.Path())
@click.option("--matcher", type=click.Choice(("auto", "exhaustive", "sequential")),
              default="auto", show_default=True)
def reconstruct(scan, matcher) -> None:
    """Structure-from-motion on the frames (one shared camera)."""
    from core import sfm

    ws = scan_or_die(scan)
    try:
        s = sfm.run_sfm(ws.sfm, single_camera=True, matcher=matcher)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Registered {s.registered_images}/{s.total_images} frames, {s.num_points:,} points, "
               f"reprojection error {s.mean_reprojection_error:.2f} px")
    if s.registered_images < max(3, s.total_images // 2):
        raise click.ClickException("fewer than half the frames registered - see README tips")
    click.echo(f"Next: scan2cad align {ws.root}")


@main.command()
@click.argument("scan", type=click.Path())
@board_options
def align(scan, **opts) -> None:
    """Scale + Z-up board frame from the ChArUco board seen in the frames."""
    from core import board

    ws = scan_or_die(scan)
    spec = make_spec(ws.read_meta().get("board"), **opts)
    try:
        res = board.calibrate(ws.sfm, spec)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    s = res["scale_mm_per_unit"]
    ws.update_meta(board=spec.to_dict(), scale_source=f"charuco/{res['method']}",
                   alignment_rms_mm=round(res["rms_mm"], 4))
    click.echo(f"Board found in {res['views']} frames; {res['method']} with {res['corners']} corners, "
               f"residual {res['rms_mm']:.3f} mm rms. Scale {s:.4f} mm/unit.")
    if res["rms_mm"] > 0.5:
        click.echo("  warning: residual > 0.5 mm - is the board flat and the square size right?")
    click.echo(f"Next: scan2cad mask {ws.root}")


@main.command()
@click.argument("scan", type=click.Path())
@click.option("--point", "points", multiple=True, help="Object click FRAME:X,Y (repeatable).")
@click.option("--neg", "negatives", multiple=True, help="Not-object click FRAME:X,Y.")
@click.option("--box", default=None, help="Box FRAME:X0,Y0,X1,Y1.")
@click.option("--model", "model_id", default=None, help="SAM 2 model id [sam2.1-hiera-tiny].")
def mask(scan, points, negatives, box, model_id) -> None:
    """Select the object (clicks/box, or interactively) and track it through all frames."""
    from .capture import masks

    ws = scan_or_die(scan)
    model_id = model_id or masks.DEFAULT_MODEL
    try:
        if points or negatives or box:
            prompt, predictor = masks.build_prompt(list(points), list(negatives), box), None
        else:
            prompt, predictor = masks.interactive_prompt(ws.frames_dir, model_id=model_id)
        res = masks.segment(ws.frames_dir, ws.masks_dir, prompt, model_id, predictor)
    except (ValueError, FileNotFoundError, KeyboardInterrupt) as exc:
        raise click.ClickException(str(exc))
    ws.update_meta(mask_prompt={"frame": prompt.frame, "points": prompt.points,
                                "labels": prompt.labels, "box": prompt.box})
    click.echo(f"Masks for {res['masked']}/{res['frames']} frames, object covers "
               f"{100 * res['mean_area_fraction']:.1f}% of the image on average.")
    if res["empty_masks"]:
        click.echo(f"  warning: empty mask in frames {res['empty_masks'][:10]} - object lost")
    click.echo(f"Next: scan2cad cloud {ws.root}")


@main.command(name="cloud")
@click.argument("scan", type=click.Path())
@click.option("--backend", type=click.Choice(("auto", "dense", "hull", "sparse")), default="auto",
              show_default=True, help="auto = dense if OpenMVS/CUDA is available, else hull.")
@click.option("--openmvs-dir", type=click.Path(), envvar="MEASURE3D_OPENMVS_DIR")
@click.option("--resolution", type=int, default=None, help="Hull voxels along the long side.")
def cloud_cmd(scan, backend, openmvs_dir, resolution) -> None:
    """Isolate the selected object as an oriented point cloud in mm."""
    from core import dense

    from .capture import cloud

    ws = scan_or_die(scan)
    if backend == "auto":
        try:
            dense.resolve_backend("auto", openmvs_dir)
            backend = "dense"
        except RuntimeError:
            backend = "hull"
            click.echo("No OpenMVS or CUDA found: using the visual hull. It overestimates top "
                       "surfaces and fills pockets; install OpenMVS for accurate geometry.")
    try:
        info = cloud.build_object_cloud(ws, backend, openmvs_dir,
                                        resolution or cloud.HULL_RESOLUTION)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    ws.update_meta(cloud_backend=backend)
    echo_json(info)
    click.echo(f"Object cloud written to {ws.object_ply}. Next: scan2cad segment {ws.root}")


# --- fitting and CAD ----------------------------------------------------------------

def harness_or_die(scan: str):
    from .agent.server import Harness

    try:
        return Harness(Path(scan))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))


@main.command(name="segment")
@click.argument("scan", type=click.Path())
@click.option("--tolerance", type=float, default=None, help="Inlier band, mm [3x noise].")
def segment_cmd(scan, tolerance) -> None:
    """Split the object cloud into faces and fit a primitive to each."""
    res = harness_or_die(scan).segment(tolerance)
    stats = res["stats"]
    echo_json(stats)
    for r in res["regions"]:
        dims = {k: r[k] for k in ("diameter", "half_angle_deg", "major_radius", "minor_radius",
                                  "offset", "nearest_axis") if k in r}
        click.echo(f"  [{r['id']:2d}] {r['kind']:9s} {r['points']:6d} pts  {dims}")
    if stats.get("noise_floor_mm"):
        click.echo(f"Noise floor (median plane rms): {stats['noise_floor_mm']:.3f} mm")


@main.command()
@click.argument("scan", type=click.Path())
@click.argument("name")
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
def build(scan, name, script) -> None:
    """Build CadQuery SCRIPT (assigning `result`) as candidate NAME."""
    echo_json(harness_or_die(scan).build(name, Path(script).read_text(encoding="utf-8")))


@main.command()
@click.argument("scan", type=click.Path())
@click.argument("name")
@click.option("--no-silhouettes", is_flag=True)
def score(scan, name, no_silhouettes) -> None:
    """Score candidate NAME against the scan."""
    try:
        echo_json(harness_or_die(scan).score(name, not no_silhouettes))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))


@main.command(name="render")
@click.argument("scan", type=click.Path())
@click.argument("name", required=False)
@click.option("--view", "views", multiple=True, help="iso, iso_back, top, front, ... (repeatable)")
def render_cmd(scan, name, views) -> None:
    """Render candidate NAME (or the bare scan) with deviation colors to <scan>/renders."""
    h = harness_or_die(scan)
    h.render(name, list(views) or None)
    click.echo(f"Wrote {h.ws.renders_dir / ((name or 'scan') + '.png')}")


@main.command()
@click.argument("scan", type=click.Path())
@click.argument("name")
@click.argument("new_name")
@click.option("--tolerance", type=float, default=None)
def refit(scan, name, new_name, tolerance) -> None:
    """Replace spline faces of candidate NAME with analytic surfaces -> NEW_NAME."""
    echo_json(harness_or_die(scan).refit(name, new_name, tolerance))


@main.group()
def dualbrep() -> None:
    """DualBrep freeform/fallback backend (needs a CUDA machine with DualBrep installed)."""


@dualbrep.command(name="prepare")
@click.argument("scan", type=click.Path())
@click.option("--bundle", type=click.Path(), default=None, help="[<scan>/dualbrep]")
def dualbrep_prepare(scan, bundle) -> None:
    """Write the normalized input bundle to run DualBrep on another machine."""
    from .cad import dualbrep as db
    from .capture import cloud

    ws = scan_or_die(scan)
    pts, nrm = cloud.load_cloud(ws.object_ply)
    b = db.prepare(pts, nrm, Path(bundle) if bundle else ws.root / "dualbrep")
    click.echo(f"Bundle ready in {b} - see {b / 'README.txt'} for the commands to run.")


@dualbrep.command(name="run")
@click.argument("scan", type=click.Path())
@click.option("--dualbrep-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--python", "python_exe", required=True, help="Python of the DualBrep conda env.")
@click.option("--name", default="dualbrep", show_default=True, help="Candidate name.")
def dualbrep_run(scan, dualbrep_dir, python_exe, name) -> None:
    """Run DualBrep locally and register the result (and its analytic refit) as candidates."""
    from .cad import dualbrep as db
    from .capture import cloud

    ws = scan_or_die(scan)
    pts, nrm = cloud.load_cloud(ws.object_ply)
    bundle = db.prepare(pts, nrm, ws.root / "dualbrep")
    try:
        db.run(bundle, Path(dualbrep_dir), python_exe)
    except (RuntimeError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))
    _register_dualbrep(ws, bundle, name)


@dualbrep.command(name="import")
@click.argument("scan", type=click.Path())
@click.option("--bundle", type=click.Path(exists=True), default=None, help="[<scan>/dualbrep]")
@click.option("--step", type=click.Path(exists=True), default=None, help="[<bundle>/scan.step]")
@click.option("--name", default="dualbrep", show_default=True, help="Candidate name.")
def dualbrep_import(scan, bundle, step, name) -> None:
    """Import a DualBrep STEP from another machine back into mm as a candidate."""
    ws = scan_or_die(scan)
    _register_dualbrep(ws, Path(bundle) if bundle else ws.root / "dualbrep",
                       name, Path(step) if step else None)


def _register_dualbrep(ws, bundle: Path, name: str, step: Path | None = None) -> None:
    from . import cad
    from .cad import dualbrep as db

    try:
        shape = db.import_step(bundle, step)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    target = ws.cad_dir / f"{name}.step"
    cad.export_step(shape, target)
    cad.export_stl(shape, target.with_suffix(".stl"))
    click.echo(f"Candidate {name}: {cad.shape_stats(shape)}")
    h = harness_or_die(str(ws.root))
    click.echo(f"Refit to analytic faces -> {name}_refit:")
    echo_json(h.refit(name, f"{name}_refit"))
    click.echo(f"Score both: scan2cad score {ws.root} {name}; scan2cad score {ws.root} {name}_refit")


# --- agents -------------------------------------------------------------------------

@main.command()
@click.argument("scan", type=click.Path())
def serve(scan) -> None:
    """Run the MCP server (stdio) for SCAN."""
    from .agent.server import serve as run_server

    scan_or_die(scan)
    run_server(Path(scan))


@main.command(name="agent")
@click.argument("scan", type=click.Path())
@click.option("--runner", type=click.Choice(("claude", "codex")), default="claude", show_default=True)
@click.option("--model", default=None, help="Model id (claude default: claude-opus-5-5; codex: its default).")
@click.option("--max-turns", type=int, default=60, show_default=True, help="Claude only.")
def agent_cmd(scan, runner, model, max_turns) -> None:
    """Let a model reverse-engineer SCAN through the MCP tools."""
    from .agent import runners

    ws = scan_or_die(scan)
    if runner == "claude":
        out = runners.run_claude(ws.root, model or runners.DEFAULT_CLAUDE_MODEL, max_turns)
    else:
        out = runners.run_codex(ws.root, model)
    click.echo(out.get("final_text", ""))
    click.echo(f"\nFinal candidate: {out['final_candidate']}  score: {out.get('final_score')}")
    click.echo(f"Run log: {out['log']}")


@main.command(name="bench")
@click.option("--part", "parts", multiple=True, help="Parts (repeatable) [all].")
@click.option("--runner", "runners", multiple=True, default=("claude",), show_default=True,
              help="claude[:MODEL] or codex[:MODEL] (repeatable).")
@click.option("--out", type=click.Path(), default="output/bench", show_default=True)
@click.option("--noise", type=float, default=0.1, show_default=True, help="Scan noise sigma, mm.")
def bench_cmd(parts, runners, out, noise) -> None:
    """Run agents on ground-truth parts and score them against the true CAD."""
    from .agent import bench
    from .synthetic.parts import PARTS

    parsed = [(r.split(":", 1)[0], r.split(":", 1)[1] if ":" in r else None) for r in runners]
    res = bench.run(list(parts) or list(PARTS), parsed, Path(out), noise)
    click.echo(bench.markdown_table(res["rows"]))


if __name__ == "__main__":
    main()
