"""Dense reconstruction backends.

Both backends consume the pycolmap sparse output and produce the same file,
<photos>/.measure3d/dense/scene_dense.ply, so meshing is backend-agnostic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from measure3d.project import Workspace

OPENMVS_BINARIES = ("InterfaceCOLMAP", "DensifyPointCloud")
OPENMVS_DIR_ENV = "MEASURE3D_OPENMVS_DIR"

OPENMVS_INSTALL_HELP = f"""\
OpenMVS binaries (InterfaceCOLMAP, DensifyPointCloud) not found.

To install on Windows:
  1. Download OpenMVS_Windows_x64.zip from https://github.com/cdcseacave/openMVS/releases
  2. Extract it, then point mesh_gen at the folder containing DensifyPointCloud.exe:
     add it to PATH, set {OPENMVS_DIR_ENV}, or pass --openmvs-dir.
On Linux, build from source: https://github.com/cdcseacave/openMVS/wiki/Building"""

GPU_UNAVAILABLE_HELP = """\
The gpu backend needs a CUDA-enabled pycolmap build; this installation reports no CUDA
support (pip wheels for Windows/macOS are CPU-only). Use --backend cpu (OpenMVS) here,
or run the export on a CUDA machine - see "Deploying on a GPU server" in the README."""


def pycolmap_has_cuda() -> bool:
    import pycolmap

    return bool(pycolmap.has_cuda)


def find_openmvs(openmvs_dir: str | None) -> dict[str, str] | None:
    """Full paths to the OpenMVS binaries, or None if any is missing.

    Search order: --openmvs-dir, the MEASURE3D_OPENMVS_DIR env var, then PATH.
    """
    search = openmvs_dir or os.environ.get(OPENMVS_DIR_ENV) or None
    tools = {}
    for name in OPENMVS_BINARIES:
        exe = shutil.which(name, path=search)
        if exe is None:
            return None
        tools[name] = exe
    return tools


def resolve_backend(requested: str, openmvs_dir: str | None) -> str:
    """Resolve 'auto' and fail fast (before any reconstruction work) if unavailable."""
    backend = requested
    if backend == "auto":
        backend = "gpu" if pycolmap_has_cuda() else "cpu"
        print(f"Backend: auto -> {backend}")
    if backend == "gpu" and not pycolmap_has_cuda():
        raise RuntimeError(GPU_UNAVAILABLE_HELP)
    if backend == "cpu" and find_openmvs(openmvs_dir) is None:
        raise RuntimeError(OPENMVS_INSTALL_HELP)
    return backend


def densify(ws: Workspace, backend: str, openmvs_dir: str | None = None) -> Path:
    """Produce (or reuse) the dense point cloud for this workspace."""
    if ws.dense_ply.exists():
        print(f"Reusing cached dense cloud: {ws.dense_ply}")
        return ws.dense_ply

    _undistort(ws)
    if backend == "gpu":
        _densify_pycolmap_cuda(ws)
    else:
        _densify_openmvs(ws, openmvs_dir)

    if not ws.dense_ply.exists():
        raise RuntimeError(f"dense reconstruction finished but {ws.dense_ply} was not produced")
    return ws.dense_ply


def _undistort(ws: Workspace) -> None:
    import pycolmap

    print("Undistorting images...")
    ws.dense_dir.mkdir(parents=True, exist_ok=True)
    pycolmap.undistort_images(ws.dense_dir, ws.sparse_model_dir, ws.photos_dir)


def _densify_openmvs(ws: Workspace, openmvs_dir: str | None) -> None:
    tools = find_openmvs(openmvs_dir)
    if tools is None:
        raise RuntimeError(OPENMVS_INSTALL_HELP)
    print(
        "Running OpenMVS dense reconstruction on CPU - this typically takes tens of "
        "minutes; progress is shown below."
    )
    scene = ws.dense_dir / "scene.mvs"
    _run(
        [
            tools["InterfaceCOLMAP"],
            "-i", str(ws.dense_dir),
            "-o", str(scene),
            "--image-folder", str(ws.dense_dir / "images"),
        ],
        cwd=ws.dense_dir,
    )
    # Writes scene_dense.mvs + scene_dense.ply into the working directory.
    _run([tools["DensifyPointCloud"], scene.name], cwd=ws.dense_dir)


def _densify_pycolmap_cuda(ws: Workspace) -> None:
    import pycolmap

    print("Running patch-match stereo (GPU)...")
    pycolmap.patch_match_stereo(ws.dense_dir)
    print("Fusing depth maps...")
    pycolmap.stereo_fusion(ws.dense_ply, ws.dense_dir)


def _run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)  # output streams to the console
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{Path(cmd[0]).stem} failed with exit code {exc.returncode}") from exc
