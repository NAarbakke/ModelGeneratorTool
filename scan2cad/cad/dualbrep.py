"""DualBrep (Autodesk AI Lab, SIGGRAPH 2026) as an optional freeform/fallback backend.

https://github.com/AutodeskAILab/DualBrep - Apache-2.0. Point cloud -> watertight
B-rep (STEP) with prismatic and free-form faces, all as splines.

It needs a CUDA GPU (torch cu128, torch_cluster) and its own conda environment
with pythonocc-core, which can't share a process with CadQuery's OCP. So it runs
as a subprocess, either here (`run`, given that environment) or on another
machine (`prepare` a bundle, run DualBrep there, `import` the STEP back).

Input contract (from its sample data): an oriented .ply with 8192 points,
centered and uniformly scaled so the longest axis spans [-0.9, 0.9]. Its output
is in those normalized coordinates; `import_step` maps it back to mm.
Like any model trained on clean CAD-derived clouds, expect it to degrade on
photogrammetry noise: check the score before trusting the result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

NUM_POINTS = 8192
HALF_EXTENT = 0.9
SHAPE_NAME = "scan"


def normalization(points: np.ndarray) -> tuple[np.ndarray, float]:
    """(center, scale) with normalized = (p - center) * scale."""
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    return center, 2 * HALF_EXTENT / float(np.max(hi - lo))


def prepare(points: np.ndarray, normals: np.ndarray, bundle: Path, seed: int = 0) -> Path:
    """Write <bundle>/sample/scan.ply, the name list and the normalization sidecar."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), min(NUM_POINTS, len(points)), replace=False)
    center, scale = normalization(points)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((points[idx] - center) * scale))
    pcd.normals = o3d.utility.Vector3dVector(normals[idx])
    (bundle / "sample").mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(bundle / "sample" / f"{SHAPE_NAME}.ply"), pcd, write_ascii=False)
    (bundle / "names.txt").write_text(SHAPE_NAME + "\n")
    (bundle / "normalization.json").write_text(json.dumps(
        {"center_mm": center.tolist(), "scale_per_mm": scale, "points": int(len(idx))}, indent=2))
    (bundle / "README.txt").write_text(_RUN_INSTRUCTIONS.format(bundle=bundle.name))
    return bundle


_RUN_INSTRUCTIONS = """\
Run DualBrep on this bundle (Linux + CUDA GPU, DualBrep repo with checkpoints/):

  cd /path/to/DualBrep && conda activate Dualbrep
  python ae_reconstruct.py config=config_pc.yaml dataset.data_root=<bundle>/sample \\
      dataset.name_list=<bundle>/names.txt runtime.output_dir=<bundle>/out/recon
  python rebuild.py --input <bundle>/out/recon --out <bundle>/out/brep
  python postprocess.py --input <bundle>/out/brep

Then copy {bundle}/out/brep/scan.step back next to normalization.json and run:
  scan2cad dualbrep import <scan> --bundle <bundle>
"""


def run(bundle: Path, dualbrep_dir: Path, python: str, rotations: str = "all") -> Path:
    """Run the three DualBrep stages in its own environment; returns the output STEP."""
    if not (dualbrep_dir / "ae_reconstruct.py").exists():
        raise FileNotFoundError(f"not a DualBrep checkout: {dualbrep_dir}")
    if not (dualbrep_dir / "checkpoints" / "pc_vae.ckpt").exists():
        raise FileNotFoundError(
            f"DualBrep checkpoints missing in {dualbrep_dir / 'checkpoints'} - see its README "
            "(hf download ADSKAILab/DualBrep checkpoints.zip)")
    b = bundle.resolve()
    recon, brep = b / "out" / "recon", b / "out" / "brep"
    stages = [
        [python, "ae_reconstruct.py", "config=config_pc.yaml", f"dataset.data_root={b / 'sample'}",
         f"dataset.name_list={b / 'names.txt'}", "runtime.compute_clustering=true",
         f"runtime.output_dir={recon}"],
        [python, "rebuild.py", "--input", str(recon), "--out", str(brep), "--rotations", rotations],
        [python, "postprocess.py", "--input", str(brep)],
    ]
    for cmd in stages:
        print("$ " + " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=dualbrep_dir, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"DualBrep stage {cmd[1]} failed (exit {exc.returncode})") from exc
    step = brep / f"{SHAPE_NAME}.step"
    if not step.exists():
        raise RuntimeError(f"DualBrep finished without a solid for this scan ({step} missing) - "
                           "it seals 85 of 100 clean test shapes; noisy scans fail more often")
    shutil.copy2(step, b / f"{SHAPE_NAME}.step")
    return b / f"{SHAPE_NAME}.step"


def import_step(bundle: Path, step: Path | None = None):
    """DualBrep STEP (normalized coords) -> shape in mm, board frame."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec

    from .. import cad

    norm = json.loads((bundle / "normalization.json").read_text())
    step = step or bundle / f"{SHAPE_NAME}.step"
    if not step.exists():
        raise FileNotFoundError(f"no DualBrep result at {step}")
    shape = cad.load_step(step)
    scale = gp_Trsf()
    scale.SetScaleFactor(1.0 / norm["scale_per_mm"])
    move = gp_Trsf()
    move.SetTranslation(gp_Vec(*norm["center_mm"]))
    move.Multiply(scale)  # p_mm = p_norm / scale + center
    return BRepBuilderAPI_Transform(shape, move, True).Shape()
