# ModelGeneratorTool

Measure the real-world dimensions of an object from photos, export a true-to-scale 3D mesh, or reverse-engineer a part from video into parametric CAD.

Four command-line tools in one repo:

- **`measure3d`** — structure-from-motion (pycolmap), scale calibration, and interactive measuring in 3D. Needs 10+ overlapping photos.
- **`measure2d`** — quick in-plane measurements from a **single photo** with a planar reference (A4 sheet, credit card, ...). 2D only: no heights, no meshes.
- **`mesh_gen`** — dense reconstruction (OpenMVS on CPU or pycolmap on CUDA), Poisson meshing, and STL/OBJ/PLY export.
- **`scan2cad`** — **video → parametric CAD (STEP)**: frames, SfM, ChArUco scale, SAM 2 object selection, primitive fitting, and an MCP tool server so a model (Claude, Codex, ...) writes and checks the CadQuery. See [scan2cad](#scan2cad-video--cad).

All four are thin CLIs over a shared **`core`** package: workspace layout, SfM, the scale/ChArUco calibration, dense backends and point picking. A calibration made by one tool is used by the others (see [Layout](#layout)).

> **All real-world lengths are millimeters** — measurements, bounding boxes, and exported meshes. CAD imports are unit-sensitive: when importing into SOLIDWORKS etc., set the import units to mm.

## Install

Requires Python 3.11–3.12 (Open3D has no 3.13 wheels yet) and [uv](https://docs.astral.sh/uv/) (or any way to get a 3.12 venv).

```powershell
uv venv --python 3.12
uv pip install --python .venv -e .                # measure3d, measure2d, mesh_gen
uv pip install --python .venv -e ".[scan2cad]"    # + scan2cad (SciPy, CadQuery/OCCT, MCP, ...)
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
#    ...or, if the object stands on a printed ChArUco board (`scan2cad board`), no clicking:
#    scale plus a Z-up frame on the board, so bounding boxes and meshes come out board-aligned.
measure3d calibrate --board --square-mm 25

# 3. Measure: pick point pairs in the viewer, get distances in mm.
#    --crop first opens a crop window (K to lock, drag a box, C to crop, Q when done)
#    so the bounding box covers only the object.
measure3d measure --crop

# 4. Export a true-scale mesh (dense reconstruction runs on first export; slow on CPU).
mesh_gen export --format stl
```

The calibration is stored in `<photos>/.measure3d/scale.json` (a similarity transform from the SfM world to mm, plus how it was made), so re-measuring and exporting never require re-calibrating.

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

## scan2cad: video → CAD

Film a part standing on a printed ChArUco board, click the part once, and get a true-scale, Z-up point cloud of just that part, its faces fitted as planes/cylinders/cones/spheres/tori, and a model-written CadQuery model scored against the scan.

```
video ─► frames ─► SfM ─► ChArUco align ─► SAM 2 masks ─► object cloud ─► segment + fit ─► agent (MCP) ─► STEP
         OpenCV    pycolmap  mm, Z up       one click      OpenMVS/hull    RANSAC + LSQ    CadQuery/OCP
```

### Install

```powershell
# SAM 2 (object masks): CPU PyTorch first, then the extras
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision
uv pip install --python .venv -e ".[scan2cad,sam,agent]"
```

OpenMVS (see [above](#openmvs-needed-for-mesh_gen-export-on-machines-without-cuda)) gives the accurate dense cloud; without it `scan2cad cloud` falls back to the visual hull (see limits below). The SAM 2.1 tiny checkpoint downloads from Hugging Face on first use.

### Capture

- `scan2cad board` writes `output/board.png` (9×7 squares of 25 mm, fits A4). Print at **100 %**, glue it flat, **measure one square** and pass `--square-mm` to `align` if it differs.
- Stand the part on the board. Matte, textured surfaces reconstruct; shiny or plain ones (machined metal, white plastic) need scanning spray or a light dusting of powder.
- Walk **one slow, continuous orbit** (no cuts), at two or three heights, **including a low one**: the camera elevation decides what the cloud can see (top faces need high views, the walls of a hole need steep ones). Keep the board in view in most frames.

### Run

```powershell
scan2cad frames input/videos/part.mp4          # -> output/scans/part (sharpest frame per time bin; longest shot if the video has cuts)
scan2cad reconstruct output/scans/part
scan2cad align output/scans/part               # ChArUco: scale + board frame (same as measure3d calibrate --board)
scan2cad mask output/scans/part                # click the part (L = part, R = not part, Enter); or --point 0:X,Y
scan2cad cloud output/scans/part               # dense if OpenMVS/CUDA is available, else visual hull
scan2cad segment output/scans/part             # faces + fitted primitives; prints the scan's noise floor
scan2cad agent output/scans/part --runner claude          # or --runner codex [--model ...]
```

Without a board, `measure3d calibrate output/scans/part/frames --known-distance <mm>` sets scale from two picked points (no Z-up frame then).

Working by hand instead of with an agent: `scan2cad build SCAN NAME model.py` (CadQuery assigning `result`), `scan2cad score SCAN NAME`, `scan2cad render SCAN NAME` (deviation-colored previews in `<scan>/renders/`).

### The agent layer

`scan2cad serve SCAN` is an MCP server (stdio) with these tools: `scan_info`, `segment_scan`, `list_regions`, `region`, `fit_region`, `build_cad`, `score_cad`, `render_cad`, `show_code`, `list_candidates`, `refit_faces`, `export_cad`. `scan2cad agent` runs it with either runner, logs each run to `<scan>/agent_runs/`, and gives the model no other tools (Claude: built-in tools off; Codex: read-only sandbox). To use it from your own session instead:

```powershell
claude mcp add scan2cad -- C:\path\to\.venv\Scripts\python.exe -m scan2cad.cli serve C:\path\to\output\scans\part
codex mcp add scan2cad -- C:\path\to\.venv\Scripts\python.exe -m scan2cad.cli serve C:\path\to\output\scans\part
```

Scores the agent sees: `scan_to_cad` (scan points the model doesn't explain: missing features), `cad_to_scan` (model surface no scan supports, counted only where ≥ 3 cameras saw it within 60° of head-on: invented features), per-region deviations, the worst areas by location, silhouette IoU against the SAM 2 masks, and an F-score at max(2.5 × tolerance, 3 × point spacing).

### Comparing models

Scores against a noisy scan reward overfitting, so `scan2cad bench` judges each run against the **true CAD** of known parts (`scan2cad/synthetic/parts.py`): volume IoU, chamfer distance, bbox error and whether the face types match.

```powershell
scan2cad bench --runner claude:claude-opus-5-5 --runner codex:<model> --out output/bench
```

Each runner gets a fresh copy of each case; results land in `output/bench/results-*.md`. `scan2cad synth PART OUT` renders a full synthetic capture (video + ground truth) to test the whole photo pipeline the same way.

### DualBrep (freeform fallback)

[DualBrep](https://github.com/AutodeskAILab/DualBrep) (Apache-2.0) turns an oriented cloud into a watertight B-rep with spline faces. It needs CUDA and its own conda env, so it runs as a subprocess or on another machine: `scan2cad dualbrep prepare SCAN` writes a normalized bundle with the commands to run, `scan2cad dualbrep import SCAN` maps the STEP back to mm. `run` does both locally when given `--dualbrep-dir` and `--python`. Import automatically adds a `<name>_refit` candidate whose faces that fit an analytic surface within tolerance are rebuilt as true planes/cylinders/cones/spheres/tori; each face swap is checked and rolled back if the solid breaks.

### Measured accuracy

Synthetic capture of `block_hole` (80 × 50 × 30 mm, 4 × R6 fillets, Ø20 through-hole), 60 frames at 960×540, run through the real pipeline on CPU:

| Stage | Result |
|---|---|
| SfM | 60/60 frames, 0.50 px reprojection, focal 819.5 (true 820) |
| ChArUco align | corner fit 0.026 mm rms; recovered camera positions within 0.18 mm mean / 0.35 mm max of truth |
| SAM 2 masks (one click) | IoU 0.995 mean, 0.992 worst frame vs exact masks |
| Dense cloud (OpenMVS) | 0.087 mm mean / 0.27 mm p95 from the true surface; extent 80.7 × 50.8 × 29.7 mm; 91 % of the visible surface covered |
| Fitted planes | within 0.05 mm of true offsets; noise floor 0.061 mm |
| Fitted cylinders | fillet Ø11.74 (true 12); hole Ø19.46 (true 20, only its top seen) |

Real footage adds lens blur, rolling shutter, compression, lighting changes and shine, so expect several times these errors, and check `noise_floor_mm` on your own scans.

### Limits

- **CPU timings** (this machine, 60 frames): SfM ~9 min, SAM 2 ~10 s/frame, OpenMVS dense ~50 min, segmentation ~3–4 min. A CUDA GPU speeds up SAM 2 and dense by an order of magnitude.
- **Visual hull** (no OpenMVS): silhouettes only, so top faces bulge by ≈ half-width × tan(lowest camera elevation) (9 mm on the test part) and holes and pockets nobody looked straight into stay filled. Fine for outlines, wrong for heights.
- **Face refit** keeps each face's existing edges and does not re-intersect neighbouring surfaces. Faces that are analytic up to noise convert cleanly (all faces of 3 of the 4 test parts); a dome with a pole can stay a spline.
- **DualBrep** is wired up but untested here (no CUDA GPU on this machine). It was trained on clean CAD-derived clouds, so score its output before trusting it.
- Narrow bands are geometrically ambiguous (a 3 mm chamfer fits a cone or a sphere equally well); such regions list the alternatives under `also_fits`.

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
.venv\Scripts\python.exe -m pytest tests -m "not slow"   # fast: math, CAD kernel, scoring, harness (~1 min)
.venv\Scripts\python.exe -m pytest tests -m slow         # segmentation + B-rep refit end to end (minutes)
```

### Layout

```
core/                 shared by every tool (numpy, OpenCV, pycolmap, Open3D, click)
  workspace.py        <images>/.measure3d/ layout, input/ and output/ dirs
  sfm.py              structure-from-motion (pycolmap)
  scale.py            measurement math, the scale.json calibration (world -> mm)
  board.py            ChArUco board: print, detect, align + calibrate a workspace
  geometry.py         Umeyama similarity, triangulation, transforms
  dense.py            dense backends: OpenMVS (CPU), pycolmap patch-match (CUDA)
  picking.py          Open3D point picking and cropping
  cliutil.py          shared click helpers (workspace lookup, board options)
measure3d/cli.py      reconstruct, calibrate (points or --board), measure
measure2d/            single-photo homography measuring (its own OpenCV UI)
mesh_gen/             Poisson meshing + export (cli.py, meshing.py)
scan2cad/             video -> CAD; needs the `scan2cad` extra
  cli.py, workspace.py   commands, per-scan folder layout
  capture/            frames from video, SAM 2 masks, object cloud
  geometry/           primitive fitting, segmentation, ray casting
  cad/                CadQuery/OCP helpers, sandboxed runner, analytic refit, DualBrep bridge
  evaluation/         CAD-vs-scan scoring, deviation renders
  agent/              MCP server (Harness), Claude/Codex runners, benchmark
  synthetic/          known test parts, synthetic captures with ground truth
tests/                test_core, test_measure2d, test_mesh_gen, test_primitives, test_scan2cad(_slow)
```

Dependencies only point downwards: the four tools import `core`, never each other.
