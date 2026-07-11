# measure3d + measure2d + mesh_gen

Measure the real-world dimensions of an object from photos, and export a true-to-scale 3D mesh.

- **`measure3d`** — structure-from-motion (pycolmap), scale calibration, and interactive measuring in 3D. Needs 10+ overlapping photos.
- **`measure2d`** — quick in-plane measurements from a **single photo** with a planar reference (A4 sheet, credit card, ...). 2D only: no heights, no meshes.
- **`mesh_gen`** — dense reconstruction (OpenMVS on CPU or pycolmap on CUDA), Poisson meshing, and STL/OBJ/PLY export.

> **All real-world lengths are millimeters** — measurements, bounding boxes, and exported meshes. CAD imports are unit-sensitive: when importing into SOLIDWORKS etc., set the import units to mm.

## Install

Requires Python 3.11–3.12 (Open3D has no 3.13 wheels yet) and [uv](https://docs.astral.sh/uv/) (or any way to get a 3.12 venv).

```powershell
uv venv --python 3.12
uv pip install --python .venv -e .
```

### OpenMVS (needed for `mesh_gen export` on machines without CUDA)

1. Download `OpenMVS_Windows_x64.zip` from <https://github.com/cdcseacave/openMVS/releases> (v2.4.0+).
2. Extract, then point `mesh_gen` at the folder containing `DensifyPointCloud.exe` — add it to `PATH`, set `MEASURE3D_OPENMVS_DIR`, or pass `--openmvs-dir`.

`measure3d` (reconstruct/calibrate/measure) works without OpenMVS — it is only needed for dense mesh export.

## Usage walkthrough

Take **10–30 overlapping photos** of the object from different angles (good light, textured surfaces, no motion blur) and drop them in `input/photos/` (every command also accepts an explicit folder argument).

```powershell
.venv\Scripts\Activate.ps1     # or prefix commands with .venv\Scripts\

# 1. Sparse reconstruction (~1-5 min). Prints images registered / points / reprojection error.
measure3d reconstruct

# 2. Calibrate scale: a viewer opens; Shift+click two points whose real distance you know
#    (e.g. the ends of a ruler lying next to the object), close with Q.
measure3d calibrate --known-distance 210

# 3. Measure: pick point pairs in the viewer, get distances in mm.
#    --crop first opens a crop window (K to lock, drag a box, C to crop, Q when done)
#    so the bounding box covers only the object.
measure3d measure --crop

# 4. Export a true-scale mesh (dense reconstruction runs on first export; slow on CPU).
mesh_gen export --format stl
```

The scale factor is stored in `<photos>/.measure3d/scale.json`, so re-measuring and exporting never require re-calibrating.

### Outputs

| Path | Contents |
|---|---|
| `<photos>/.measure3d/` | reconstruction cache: database, sparse/dense models, `scale.json` |
| `<photos>/.measure3d/exports/` | exported meshes / point clouds |
| `output/dimensions/<photoset>-measurements.json` | copy of each measuring session |
| `output/models/` | copy of each exported mesh |

### `mesh_gen export` options

| Flag | Default | Meaning |
|---|---|---|
| `--format stl\|obj\|ply\|pointcloud` | `stl` | `ply` keeps vertex color; `pointcloud` exports the scaled dense cloud (no meshing) |
| `--depth N` | `9` | Poisson octree depth — higher = more detail, more time/memory |
| `--backend auto\|cpu\|gpu` | `auto` | `auto` uses CUDA if pycolmap has it, else OpenMVS on CPU |
| `--no-crop` | off | skip the interactive crop (use for headless runs) |
| `--openmvs-dir DIR` | — | folder with the OpenMVS binaries |

The export refuses to run without calibration — an unscaled mesh is useless downstream. Expect the CPU dense pass to take **tens of minutes**; it is cached in `.measure3d/dense/` and reused by later exports.

## measure2d: single-photo measurements

When you only need length/width of something lying on a flat surface, skip photogrammetry entirely: put a reference object of known size **in the same plane** (a sheet of A4 paper, a credit card, or any rectangle you can measure) and take **one photo**.

```powershell
measure2d photo.jpg                     # A4 as reference (default)
measure2d photo.jpg --ref credit-card   # 85.6 x 53.98 mm
measure2d photo.jpg --ref 120x80        # custom rectangle, mm
measure2d photo.jpg --ref-length 300    # line mode: click 2 ruler marks 300 mm apart
```

A viewer opens on the photo. You place each point with a **two-stage click**: first click *near* the target on the fitted-to-screen image (the AIM view), then a magnified patch of the full-resolution pixels opens and you click the exact spot (the CLICK view; right-click or `b` to re-aim, `Esc` to cancel). This zoom step is what keeps accuracy high on high-resolution photos — clicking the shrunk-to-fit image directly would snap to whole screen pixels, each worth several image pixels.

First place the reference's 4 corners (top-left, top-right, bottom-right, bottom-left as seen — perspective is corrected via homography), then place point pairs; each completed pair prints its distance in mm (`u` undoes, `Enter`/`q` finishes). An annotated image and a JSON report land in `output/dimensions/`.

### Accuracy

Validated against synthetic photos rendered through a physically correct camera + lens-distortion model (see the tests). With a phone's main camera, the reference placed near what you're measuring, and both kept off the frame edges, expect roughly **0.2–1% error** (about 0.2–1 mm on a 100 mm object). The zoom click removes the click-precision penalty, so the residual error is dominated by uncorrected lens distortion, which grows toward the frame corners.

Limits to keep in mind:

- Only distances **in the reference's plane** are valid. Objects sticking up out of the plane, or closer/farther from the camera, measure wrong — that's the physics of a single view, and it's what `measure3d`'s multi-photo pipeline solves.
- Line mode (`--ref-length`) does no perspective correction — use it only for photos taken square-on; the 4-corner modes handle oblique shots.
- Lens distortion is not corrected (it needs the physical camera and a checkerboard calibration). Prefer a standard/main lens over ultra-wide or action cams, and keep the reference and target away from the frame edges where distortion is worst.

## Troubleshooting

- **"only N of M images registered"** — the photos don't overlap enough or lack texture. Add more angles with ~60-80% overlap between neighboring shots, avoid plain/reflective surfaces, improve lighting.
- **Distances look wrong** — recalibrate with a longer reference distance (longer = less relative picking error), picking well-separated, clearly identifiable points.
- **Mesh has blobs/bubbles** — crop tighter before meshing; the low-density trim removes most Poisson artifacts, but background points create surfaces of their own.

## Deploying on a GPU server

Dense reconstruction is the only slow step, and pycolmap's patch-match stereo is CUDA-only. The pip wheels for **Windows/macOS are CPU-only**; CUDA-enabled wheels exist **for Linux** as [`pycolmap-cuda12`](https://pypi.org/project/pycolmap-cuda12/). Practical setup:

- **Requirements**: Linux box with an NVIDIA GPU (≥4 GB VRAM recommended), driver supporting CUDA 12.x.
- **Verify GPU detection**: `python -c "import pycolmap; print(pycolmap.has_cuda)"` must print `True`. If it's `False`, you have the CPU wheel — install `pycolmap-cuda12` instead of `pycolmap`.
- **Expected speedup**: roughly 10–50× vs the CPU/OpenMVS path for the dense pass (minutes instead of tens of minutes), depending on GPU and image count.

### Split workflow (recommended)

Calibrate and measure **locally** (these open interactive viewers, which need a display), and run only the dense export on the server:

```bash
# local: reconstruct + calibrate as usual, then sync photos + the .measure3d cache
rsync -a ./input/photos/ server:/work/photos/     # includes .measure3d/ with scale.json

# server (headless): dense reconstruction + export at true scale
mesh_gen export /work/photos --backend gpu --no-crop --format stl

# copy the result back
rsync -a server:/work/photos/.measure3d/exports/ ./output/models/
```

`--no-crop` matters on a headless server: it skips the interactive crop window. Crop later locally (or export `--format pointcloud` and mesh locally) if the scene has a lot of background.

### CUDA Linux setup

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.12 python3.12-venv && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN python3.12 -m venv .venv \
 && .venv/bin/pip install pycolmap-cuda12 open3d numpy click \
 && .venv/bin/pip install --no-deps -e .
ENTRYPOINT [".venv/bin/mesh_gen"]
```

```bash
docker build -t mesh_gen .
docker run --gpus all -v $PWD/photos:/photos mesh_gen export /photos --backend gpu --no-crop
```

(`--no-deps` on the editable install stops pip from replacing the CUDA wheel with plain `pycolmap`, which the package metadata asks for. Equivalent bare-metal steps: `python3.12 -m venv`, then the same two pip installs.)

## Development

```powershell
.venv\Scripts\python.exe -m pytest tests   # smoke test: scaling/geometry math on synthetic data
```

Package layout: `measure3d/` (project paths, SfM, scaling math, interactive picking, CLI) and `mesh_gen/` (dense backends, meshing, CLI); `mesh_gen` reads the `.measure3d/` workspace that `measure3d` produces.
