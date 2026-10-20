"""Object selection with SAM 2: prompt one frame, propagate the mask through all frames.

Prompts are clicks (positive = object, negative = not object) and/or a box on
any frame. Propagation runs forward and backward from the prompted frame, so
the prompt doesn't have to be on frame 0. CPU works (about 1-3 s per frame with
the tiny model); a CUDA GPU is used when PyTorch sees one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"
OBJ_ID = 1


@dataclass
class Prompt:
    frame: int = 0
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)  # 1 = object, 0 = background
    box: tuple[float, float, float, float] | None = None

    def is_empty(self) -> bool:
        return not self.points and self.box is None


def parse_point(spec: str) -> tuple[int, float, float]:
    """'FRAME:X,Y' or 'X,Y' (frame 0) -> (frame, x, y)."""
    frame, _, xy = spec.rpartition(":")
    x, y = (float(v) for v in xy.split(","))
    return int(frame or 0), x, y


def parse_box(spec: str) -> tuple[int, tuple[float, float, float, float]]:
    frame, _, coords = spec.rpartition(":")
    vals = tuple(float(v) for v in coords.split(","))
    if len(vals) != 4:
        raise ValueError("box must be X0,Y0,X1,Y1")
    return int(frame or 0), vals  # type: ignore[return-value]


def build_prompt(points: list[str], negatives: list[str], box: str | None) -> Prompt:
    prompt = Prompt()
    frames = set()
    for spec, label in [(p, 1) for p in points] + [(n, 0) for n in negatives]:
        f, x, y = parse_point(spec)
        frames.add(f)
        prompt.points.append((x, y))
        prompt.labels.append(label)
    if box:
        f, prompt.box = parse_box(box)
        frames.add(f)
    if len(frames) > 1:
        raise ValueError("all clicks and the box must be on the same frame")
    prompt.frame = frames.pop() if frames else 0
    return prompt


def load_predictor(model_id: str = DEFAULT_MODEL):
    import torch
    from sam2.build_sam import build_sam2_video_predictor_hf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_sam2_video_predictor_hf(model_id, device=device)


def segment(frames_dir: Path, masks_dir: Path, prompt: Prompt,
            model_id: str = DEFAULT_MODEL, predictor=None) -> dict:
    """Propagate the prompted object through every frame; write masks_dir/<frame>.png."""
    import cv2
    import torch

    if prompt.is_empty():
        raise ValueError("no prompt: give at least one click or a box")
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frames_dir} - run `scan2cad frames` first")
    predictor = predictor or load_predictor(model_id)
    masks_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        _add_prompt(predictor, state, prompt)
        areas: dict[int, float] = {}
        for reverse in (False, True):
            for idx, _, logits in predictor.propagate_in_video(
                state, start_frame_idx=prompt.frame, reverse=reverse
            ):
                mask = (logits[0, 0] > 0).cpu().numpy()
                cv2.imwrite(str(masks_dir / (frames[idx].stem + ".png")),
                            mask.astype(np.uint8) * 255)
                areas[idx] = float(mask.mean())
            if prompt.frame == 0:
                break  # nothing before frame 0
    empty = sorted(i for i, a in areas.items() if a == 0.0)
    return {"frames": len(frames), "masked": len(areas), "empty_masks": empty,
            "mean_area_fraction": float(np.mean(list(areas.values())))}


def _add_prompt(predictor, state, prompt: Prompt):
    pts = np.asarray(prompt.points, dtype=np.float32) if prompt.points else None
    lbl = np.asarray(prompt.labels, dtype=np.int32) if prompt.points else None
    box = np.asarray(prompt.box, dtype=np.float32) if prompt.box else None
    _, _, logits = predictor.add_new_points_or_box(
        state, frame_idx=prompt.frame, obj_id=OBJ_ID, points=pts, labels=lbl, box=box
    )
    return (logits[0, 0] > 0).cpu().numpy()


def interactive_prompt(frames_dir: Path, frame: int = 0, model_id: str = DEFAULT_MODEL,
                       predictor=None) -> tuple[Prompt, object]:
    """Click on the object in an OpenCV window with a live SAM 2 preview.

    Left click = object, right click = not object, Z = undo, Enter = accept,
    A/D = previous/next frame (clears clicks), Esc = cancel.
    Returns the prompt and the loaded predictor (reuse it for `segment`).
    """
    import cv2
    import torch

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    predictor = predictor or load_predictor(model_id)
    prompt = Prompt(frame=frame)
    window = "scan2cad: click the object (L=object, R=not, Z=undo, Enter=done)"
    clicked: list[tuple[float, float, int]] = []

    def on_mouse(event, x, y, *_):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            clicked.append((x / scale, y / scale, 1 if event == cv2.EVENT_LBUTTONDOWN else 0))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    mask, shown_clicks, scale = None, -1, 1.0
    with torch.inference_mode():
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        while True:
            img = cv2.imread(str(frames[prompt.frame]))
            scale = min(1.0, 1280 / img.shape[1])
            if len(clicked) != shown_clicks:
                prompt.points = [(x, y) for x, y, _ in clicked]
                prompt.labels = [lab for _, _, lab in clicked]
                predictor.reset_state(state)
                mask = _add_prompt(predictor, state, prompt) if clicked else None
                shown_clicks = len(clicked)
            view = img.copy()
            if mask is not None:
                view[mask] = (0.5 * view[mask] + [0, 110, 0]).astype(np.uint8)
            for x, y, lab in clicked:
                cv2.circle(view, (int(x), int(y)), 6, (0, 255, 0) if lab else (0, 0, 255), -1)
            cv2.putText(view, f"frame {prompt.frame}/{len(frames) - 1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(window, cv2.resize(view, None, fx=scale, fy=scale))
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and clicked:
                break
            if key == 27:
                cv2.destroyWindow(window)
                raise KeyboardInterrupt("selection cancelled")
            if key in (ord("z"), ord("Z")) and clicked:
                clicked.pop()
            if key in (ord("a"), ord("d")):
                prompt.frame = int(np.clip(prompt.frame + (1 if key == ord("d") else -1),
                                           0, len(frames) - 1))
                clicked.clear()
                shown_clicks = -1
    cv2.destroyWindow(window)
    return prompt, predictor


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
