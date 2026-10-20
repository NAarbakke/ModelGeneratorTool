"""Benchmark: agents vs ground-truth CAD, so model comparisons don't reward overfitting the scan.

A case is a scan workspace with a known part. Two ways to make one:
  quick  sample the true surface (minus the never-seen underside) + Gaussian noise. Seconds.
  video  render a synthetic capture and run the full pipeline (frames, SfM, ChArUco,
         SAM 2, cloud). Hours on CPU, but exercises everything.

Each runner gets its own copy of the case. The final candidate is judged against
the true part, not the scan: volume IoU, two-way surface distance, bbox error and
whether the face types match (a correct block_hole is 6 planes + 5 cylinders,
not a pile of splines).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import cad
from ..evaluation import scoring
from ..synthetic.parts import PARTS
from ..workspace import ScanWorkspace

FLOOR_Z = 0.75


def make_quick_case(part: str, out: Path, noise_mm: float = 0.1, points: int = 60000,
                    seed: int = 0) -> ScanWorkspace:
    import open3d as o3d

    if part not in PARTS:
        raise KeyError(f"unknown part {part!r}; choose from {', '.join(PARTS)}")
    ws = ScanWorkspace(out)
    gt_dir = out / "gt"
    res = cad.run_script(PARTS[part], gt_dir, "part")
    if not res.get("ok"):
        raise RuntimeError(res["error"])
    mesh = scoring.CadMesh.from_shape(cad.load_step(gt_dir / "part.step"), 0.01)
    pts, nrm = mesh.sample(points, seed=seed)
    keep = pts[:, 2] > FLOOR_Z
    rng = np.random.default_rng(seed)
    pts = pts[keep] + rng.normal(0, noise_mm, (keep.sum(), 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.normals = o3d.utility.Vector3dVector(nrm[keep])
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ws.object_ply), pcd)
    ws.update_meta(bench_part=part, noise_mm=noise_mm, cloud_backend="synthetic-sampled",
                   scale_source="ground truth")
    return ws


def evaluate(candidate_step: Path, gt_step: Path) -> dict:
    """Judge a candidate against the true part."""
    from scipy.spatial import cKDTree

    cand, gt = cad.load_step(candidate_step), cad.load_step(gt_step)
    cs, gs = cad.shape_stats(cand), cad.shape_stats(gt)
    cm, gm = scoring.CadMesh.from_shape(cand, 0.02), scoring.CadMesh.from_shape(gt, 0.02)
    cp, _ = cm.sample(20000)
    gp, _ = gm.sample(20000)
    d_cg, _ = cKDTree(gp).query(cp)
    d_gc, _ = cKDTree(cp).query(gp)
    faces_ok = cs["faces"] == gs["faces"]
    return {
        "valid": cs["valid"] and cs["solids"] == 1,
        "volume_iou": round(cad.volume_iou(cand, gt), 4),
        "chamfer_mean_mm": round(float((d_cg.mean() + d_gc.mean()) / 2), 4),
        "hausdorff_mm": round(float(max(d_cg.max(), d_gc.max())), 3),
        "bbox_error_mm": [round(a - b, 3) for a, b in zip(cs["bbox_size_mm"], gs["bbox_size_mm"])],
        "faces": cs["faces"], "gt_faces": gs["faces"], "face_types_match": faces_ok,
    }


def run(parts: list[str], runners: list[tuple[str, str | None]], out: Path,
        noise_mm: float = 0.1) -> dict:
    """Every runner on every part; returns and writes a results table."""
    from . import runners

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in parts:
        base = out / "cases" / part
        if not (base / "object.ply").exists():
            make_quick_case(part, base, noise_mm)
        for runner, model in runners:
            label = f"{runner}-{model or 'default'}".replace("/", "_")
            case = out / "runs" / label / part
            if case.exists():
                shutil.rmtree(case)
            shutil.copytree(base, case)
            try:
                if runner == "claude":
                    log = runners.run_claude(case, model or runners.DEFAULT_CLAUDE_MODEL)
                elif runner == "codex":
                    log = runners.run_codex(case, model)
                else:
                    raise ValueError(f"unknown runner {runner!r}")
            except Exception as exc:  # one failed run shouldn't sink the table
                rows.append({"part": part, "runner": label, "error": repr(exc)})
                continue
            row = {"part": part, "runner": label, "seconds": log["seconds"],
                   "cost_usd": log.get("cost_usd"), "final": log["final_candidate"],
                   "candidates": len(log["candidates"]), "log": log["log"]}
            step = ScanWorkspace(case).cad_dir / f"{log['final_candidate']}.step"
            if log["final_candidate"] and step.exists():
                row.update(evaluate(step, base / "gt" / "part.step"))
            else:
                row["error"] = "no final candidate"
            rows.append(row)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = {"noise_mm": noise_mm, "rows": rows}
    (out / f"results-{stamp}.json").write_text(json.dumps(result, indent=1))
    (out / f"results-{stamp}.md").write_text(markdown_table(rows))
    return result


def markdown_table(rows: list[dict]) -> str:
    head = "| part | runner | valid | volume IoU | chamfer mm | faces match | time s | cost $ |\n"
    head += "|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        if "error" in r and "volume_iou" not in r:
            lines.append(f"| {r['part']} | {r['runner']} | error: {r['error'][:60]} | | | | | |")
            continue
        cost = f"{r['cost_usd']:.2f}" if r.get("cost_usd") is not None else "-"
        lines.append(f"| {r['part']} | {r['runner']} | {r['valid']} | {r['volume_iou']:.4f} | "
                     f"{r['chamfer_mean_mm']:.3f} | {r['face_types_match']} | {r['seconds']:.0f} | {cost} |")
    return head + "\n".join(lines) + "\n"
