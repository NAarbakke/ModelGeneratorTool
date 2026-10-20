"""Video -> sharp, evenly spread frames for SfM.

One pass scores every frame (sharpness = variance of the Laplacian; cut score =
color-histogram change from the previous frame). The video is split into shots
at hard cuts, the longest shot is used unless a time range is given, and the
sharpest frame of each of N equal time bins is kept. Frames far blurrier than
the shot's median are dropped even if they were the best of their bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANALYSIS_WIDTH = 480  # scoring runs on a downscaled copy; full-res frames are written
CUT_THRESHOLD = 0.5   # Bhattacharyya distance between consecutive HSV histograms
MIN_SHARPNESS_RATIO = 0.35


@dataclass
class FrameScores:
    fps: float
    sharpness: np.ndarray
    cut_score: np.ndarray  # cut_score[i]: change from frame i-1 to i (0 for frame 0)


def score_video(video: Path) -> FrameScores:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sharp, cuts, prev_hist = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (ANALYSIS_WIDTH, round(h * ANALYSIS_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cuts.append(0.0 if prev_hist is None else
                    cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        prev_hist = hist
    cap.release()
    if not sharp:
        raise RuntimeError(f"no frames could be decoded from {video}")
    return FrameScores(fps, np.asarray(sharp), np.asarray(cuts))


def shots(cut_score: np.ndarray, threshold: float = CUT_THRESHOLD) -> list[tuple[int, int]]:
    """[start, end) frame ranges between hard cuts."""
    bounds = [0, *np.flatnonzero(cut_score > threshold).tolist(), len(cut_score)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def select_frames(sharpness: np.ndarray, start: int, end: int, count: int,
                  min_ratio: float = MIN_SHARPNESS_RATIO) -> list[int]:
    """Sharpest frame per equal-width bin of [start, end), minus outright blurry ones."""
    if end - start <= 0:
        return []
    count = min(count, end - start)
    edges = np.linspace(start, end, count + 1).round().astype(int)
    picks = [a + int(np.argmax(sharpness[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    floor = min_ratio * np.median(sharpness[start:end])
    return [i for i in picks if sharpness[i] >= floor]


def extract(video: Path, out_dir: Path, count: int = 100, start_s: float | None = None,
            end_s: float | None = None, max_width: int = 1920) -> dict:
    """Write the selected frames as 00000.jpg ... and return a metadata dict."""
    import cv2

    scores = score_video(video)
    n = len(scores.sharpness)
    if start_s is not None or end_s is not None:
        a = 0 if start_s is None else max(0, int(start_s * scores.fps))
        b = n if end_s is None else min(n, int(end_s * scores.fps))
        shot_list = shots(scores.cut_score[a:b])
        if len(shot_list) > 1:
            print(f"warning: {len(shot_list) - 1} hard cut(s) inside the chosen range - "
                  "SfM needs continuous camera motion; consider a narrower range")
    else:
        shot_list = shots(scores.cut_score)
        a, b = max(shot_list, key=lambda s: s[1] - s[0])
        if len(shot_list) > 1:
            print(f"{len(shot_list)} shots found; using the longest: "
                  f"{a / scores.fps:.1f}-{b / scores.fps:.1f} s (pass --start/--end to choose)")
    if b - a < 2:
        raise RuntimeError("the chosen range contains fewer than 2 frames")

    keep = select_frames(scores.sharpness, a, b, count)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video))
    keep_set, written, idx = set(keep), [], 0
    while idx < b:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep_set:
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, round(h * max_width / w)),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{len(written):05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(idx)
        idx += 1
    cap.release()
    return {
        "video": str(video.resolve()), "fps": scores.fps, "total_frames": n,
        "range_s": [a / scores.fps, b / scores.fps], "shots": len(shot_list),
        "source_frames": written,
        "dropped_blurry": len(select_frames(scores.sharpness, a, b, count, 0.0)) - len(keep),
    }
