"""scan2cad: video -> masked, true-scale point cloud -> fitted primitives -> CAD (STEP).

Built on the shared `core` package (SfM, ChArUco calibration, dense backends).

capture     frames from video, SAM 2 masks, the isolated object cloud
geometry    primitive fitting, face segmentation, ray casting
cad         CadQuery/OCP helpers (package root), sandboxed script runner,
            analytic refit of spline faces, DualBrep bridge
evaluation  CAD-vs-scan scoring and deviation renders
agent       MCP server (Harness), Claude/Codex runners, ground-truth benchmark
synthetic   known test parts and synthetic captures with ground truth
workspace   the per-scan folder layout
"""

__version__ = "0.1.0"
