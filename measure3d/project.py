"""Workspace layout: the .measure3d/ directory inside a photo folder, plus output dirs.

The whole .measure3d/ directory is designed to be synced to a GPU server as-is:
it carries the database, sparse model, and the scale.json calibration sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_PHOTOS_DIR = "input/photos"
# Convenience copies land here when the folders exist (they do in this project).
OUTPUT_DIMS_DIR = "output/dimensions"
OUTPUT_MODELS_DIR = "output/models"


@dataclass(frozen=True)
class Workspace:
    """All paths derived from a photo folder."""

    photos_dir: Path

    @property
    def root(self) -> Path:
        return self.photos_dir / ".measure3d"

    @property
    def database_path(self) -> Path:
        return self.root / "database.db"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse"

    @property
    def sparse_model_dir(self) -> Path:
        return self.sparse_dir / "0"

    @property
    def scale_path(self) -> Path:
        return self.root / "scale.json"

    @property
    def dense_dir(self) -> Path:
        return self.root / "dense"

    @property
    def dense_ply(self) -> Path:
        return self.dense_dir / "scene_dense.ply"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def name(self) -> str:
        """Photoset name used to label output files."""
        return self.photos_dir.resolve().name

    def list_images(self) -> list[Path]:
        return sorted(
            p for p in self.photos_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def has_sparse(self) -> bool:
        # points3D is written in binary (.bin) by default; a text model uses .txt
        return any(
            (self.sparse_model_dir / f"points3D{ext}").exists() for ext in (".bin", ".txt")
        )


def resolve_photos_dir(photos: str | None) -> Path:
    """Resolve the photos argument, defaulting to ./input/photos."""
    path = Path(photos) if photos else Path(DEFAULT_PHOTOS_DIR)
    if not path.is_dir():
        raise FileNotFoundError(f"photo folder not found: {path}")
    return path


def output_dir(kind: str) -> Path | None:
    """Return ./output/dimensions or ./output/models if it exists in the current directory."""
    path = Path(kind)
    return path if path.is_dir() else None
