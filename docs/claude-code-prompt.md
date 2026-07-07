# Prompt for Claude Code

Build a Python CLI tool called `measure3d` that computes real-world dimensions of an object from multiple photos taken at different angles.

## Core pipeline
1. **Reconstruction**: Use `pycolmap` (not the COLMAP binary via subprocess) to run structure-from-motion on a folder of images: feature extraction → matching → sparse reconstruction. Sparse is sufficient for measurement; dense is only triggered by the export command (see step 2).
2. **Dense reconstruction (for export only)**: Support two backends behind a common interface, selected by `--backend auto|cpu|gpu` (default `auto`, which detects CUDA):
   - **CPU backend (default for this machine — no GPU available)**: OpenMVS, called as subprocess via its CLI binaries (`InterfaceCOLMAP` → `DensifyPointCloud`). It consumes the pycolmap sparse output. Verify OpenMVS is on PATH at startup and give install instructions if missing. Expect slow runtimes on CPU (tens of minutes) — print progress and set expectations in the output.
   - **GPU backend**: pycolmap's dense MVS (undistortion → patch-match stereo → stereo fusion), which requires a CUDA GPU. Do not assume it works here — but implement it, gate it behind CUDA detection, and fail with a clear message if selected without CUDA.
   - Keep the backend abstraction thin: both must produce a dense point cloud file in the same location/format so the meshing step is backend-agnostic.
   Sparse remains the default for measurement (10x faster than any dense pass).
3. **Scaling**: SfM output is scale-ambiguous. Implement scale calibration:
   - Primary method: open the sparse point cloud in Open3D's interactive visualizer (`VisualizerWithEditing` or the newer GUI picking API), let me shift-click two points, then prompt me in the terminal for the real-world distance between them (in mm). Compute and store the scale factor.
   - Persist the scale factor to a JSON sidecar file next to the reconstruction so re-measuring doesn't require re-calibrating.
4. **Measurement**: After scaling, let me pick point pairs in the viewer and print the real distance for each pair. Also print the scaled axis-aligned bounding box of the object (with a warning that it includes background points unless cropped — offer an Open3D crop step).
5. **Meshing & export**: Convert the dense point cloud into a triangle mesh at true real-world scale:
   - Apply the stored scale factor to the dense cloud BEFORE meshing. Refuse to export if no calibration exists (an unscaled mesh is useless downstream) — tell the user to run `calibrate` first.
   - Offer an interactive Open3D crop step so background/floor points can be removed before meshing.
   - Estimate normals, then run Poisson surface reconstruction (Open3D). Trim low-density vertices to remove Poisson's "blob" artifacts. Expose `--depth` (default 9) as a quality knob.
   - Export formats: **STL** (3D printing / SOLIDWORKS mesh import), **OBJ**, and **PLY** (keeps color; also export the raw scaled point cloud as PLY for point-cloud workflows). Units: millimeters — state this in the output message and README, since CAD imports are unit-sensitive.
   - Print mesh stats after export: vertex/triangle count, watertight or not, bounding box in mm.

## CLI design
- `measure3d reconstruct ./photos` → runs SfM, saves reconstruction to `./photos/.measure3d/`
- `measure3d calibrate ./photos --known-distance 210` → point picking + scale
- `measure3d measure ./photos` → interactive measurement session
- `measure3d export ./photos --format stl` → dense reconstruction (if not cached) → optional crop → Poisson mesh → scaled export to `./photos/.measure3d/exports/`. Support `--format stl|obj|ply|pointcloud`, `--depth N`, and `--backend auto|cpu|gpu`.
- Clear progress output; fail with actionable messages (e.g., "only 3 images registered — add more overlapping photos or improve texture/lighting").

## Constraints and quality bar
- Python 3.11+, dependencies: pycolmap, open3d, numpy, click (or argparse). OpenMVS binaries are an external dependency for the CPU dense backend — subprocess calls are acceptable for OpenMVS only; SfM stays in pycolmap. No web framework, no GUI toolkit.
- Keep it a single small package (~4-6 modules). Type hints. A README with install steps and a usage walkthrough.
- The README must include a **"Deploying on a GPU server"** section: CUDA/driver requirements for pycolmap's patch-match stereo, how to verify GPU detection, expected speedup vs CPU, and a recommended split workflow — calibrate/measure locally (these need a display), sync the `.measure3d/` directory (it carries the scale factor), run dense reconstruction + export headless on the server with `--backend gpu`. Include a sample Dockerfile or exact apt/pip steps for a CUDA-enabled Linux box.
- Handle the common failure mode: SfM failing to register images. Detect it and report how many images registered.
- Write a smoke test that runs the geometry/scaling math (not full SfM) on synthetic data.

## How to work (token discipline)
- Use subagents for context isolation, not parallel coding:
  - Delegate dependency installation and environment verification to a subagent; have it return only "success + versions" or the specific error.
  - Delegate pycolmap API exploration (reading docs/source to confirm current function signatures — the API has changed across versions) to a subagent that returns a short cheat-sheet of the exact calls needed.
  - Delegate running the full reconstruction on my test images to a subagent; it should return only the summary stats (images registered, points, reprojection error), not the raw logs.
- Keep all core code authorship in the main agent — do not split the package's modules across subagents.
- Before writing code, confirm the installed pycolmap version and adapt to its actual API rather than assuming.

## First steps
1. Set up the environment (subagent), confirm pycolmap + open3d import cleanly.
2. Verify the pycolmap API surface (subagent).
3. Scaffold the package and implement `reconstruct`.
4. Then `calibrate`, then `measure`, testing against my images in ./photos as you go.

Ask me before making any architectural change to the above (e.g., switching libraries, adding dense reconstruction).
