"""Shared building blocks for measure3d, measure2d, mesh_gen and scan2cad.

workspace  image-folder layout (.measure3d/) and output dirs
sfm        structure-from-motion (pycolmap)
scale      calibration math and the scale.json sidecar (SfM world -> mm)
board      ChArUco board: printable target, detection, alignment
geometry   similarity transforms (Umeyama), triangulation
dense      dense reconstruction backends (OpenMVS on CPU, pycolmap on CUDA)
picking    interactive Open3D point picking and cropping
cliutil    click helpers shared by the CLIs
"""

__version__ = "0.1.0"
