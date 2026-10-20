"""Scan workspace layout: one folder per capture, holding every stage's output.

<scan>/
  scan.json           capture metadata (source video, kept frame indices, board spec)
  frames/             extracted frames 00000.jpg ... (SAM 2 needs integer-named JPEGs)
  frames/.measure3d/  the shared core workspace: SfM database, sparse/dense models, and
                      scale.json (COLMAP world -> mm; board frame when from ChArUco)
  masks/              one PNG per frame, 255 = selected object
  object.ply          the isolated object: points + normals, mm, board frame (Z up)
  regions.json/.ply   segmentation + fitted primitives
  cad/<id>.*          candidate CAD models (CadQuery source, STEP, STL, score)
  renders/            PNG previews
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import scale
from core.workspace import Workspace as SfmWorkspace

DEFAULT_SCANS_DIR = "output/scans"


@dataclass(frozen=True)
class ScanWorkspace:
    root: Path

    @property
    def meta_path(self) -> Path:
        return self.root / "scan.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def sfm(self) -> SfmWorkspace:
        return SfmWorkspace(self.frames_dir)

    @property
    def object_ply(self) -> Path:
        return self.root / "object.ply"

    @property
    def regions_json(self) -> Path:
        return self.root / "regions.json"

    @property
    def regions_ply(self) -> Path:
        return self.root / "regions.ply"

    @property
    def cad_dir(self) -> Path:
        return self.root / "cad"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def name(self) -> str:
        return self.root.resolve().name

    def frame_paths(self) -> list[Path]:
        return sorted(self.frames_dir.glob("*.jpg"))

    def mask_path(self, frame: Path | str) -> Path:
        return self.masks_dir / (Path(frame).stem + ".png")

    def calibration(self) -> scale.Calibration | None:
        return scale.load_calibration(self.sfm.scale_path)

    def world_to_mm(self) -> np.ndarray | None:
        cal = self.calibration()
        return None if cal is None else cal.world_to_mm

    def read_meta(self) -> dict:
        return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}

    def update_meta(self, **fields) -> None:
        meta = self.read_meta()
        meta.update(fields)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2))


def default_scan_dir(video: Path) -> Path:
    return Path(DEFAULT_SCANS_DIR) / video.stem


def resolve_scan(scan: str | None) -> ScanWorkspace:
    if scan is None:
        raise FileNotFoundError("no scan folder given (e.g. output/scans/<name>)")
    root = Path(scan)
    if not root.is_dir():
        raise FileNotFoundError(f"scan folder not found: {root}")
    return ScanWorkspace(root)
