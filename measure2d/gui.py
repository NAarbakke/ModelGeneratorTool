"""Interactive OpenCV click UI for measure2d.

Importing this module imports OpenCV (which needs a display) - the CLI defers
it so --help and error paths stay fast and headless-safe.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

MAX_W, MAX_H = 1400, 900  # fit the viewer window on screen; clicks map back to full-res
_GREEN = (80, 220, 80)
_YELLOW = (60, 200, 255)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Canceled(Exception):
    """The user pressed Esc or closed the window before finishing."""


def load_image(path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return image


def collect_points(image: np.ndarray, labels: list[str], window: str = "measure2d") -> np.ndarray:
    """Collect one click per label, in order. u = undo, Esc/close = cancel.

    Returns an Nx2 array in original-image pixel coordinates.
    """
    scale, display, clicks = _open(image, window)
    try:
        while len(clicks) < len(labels):
            frame = display.copy()
            for point, label in zip(clicks, labels):
                _mark(frame, point, label)
            _banner(frame, f"Click: {labels[len(clicks)]}   (u = undo, Esc = cancel)")
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or not _window_open(window):
                raise Canceled()
            if key == ord("u") and clicks:
                clicks.pop()
    finally:
        _close(window)
    return np.array(clicks, dtype=float) / scale


def measure_loop(
    image: np.ndarray, to_metric: Callable[[np.ndarray], np.ndarray], window: str = "measure2d"
) -> tuple[list[float], np.ndarray]:
    """Click point pairs; each completed pair is measured and printed live.

    u = undo, Enter/q/Esc/close = finish. Returns the distances (mm) and a
    full-resolution annotated copy of the image.
    """
    scale, display, clicks = _open(image, window)
    printed = 0
    try:
        while True:
            pairs = [(clicks[i], clicks[i + 1]) for i in range(0, len(clicks) - 1, 2)]
            distances = [_pair_mm(to_metric, a, b, scale) for a, b in pairs]

            printed = min(printed, len(distances))
            while printed < len(distances):
                print(f"  measurement {printed + 1}: {distances[printed]:.2f} mm")
                printed += 1

            frame = display.copy()
            for (a, b), dist in zip(pairs, distances):
                _segment(frame, a, b, f"{dist:.1f} mm", thickness=2, font_scale=0.6)
            if len(clicks) % 2:
                cv2.circle(frame, clicks[-1], 5, _YELLOW, -1)
            _banner(frame, f"{len(distances)} measured - click point pairs   (u = undo, Enter/q = done)")
            cv2.imshow(window, frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (27, 13, ord("q")) or not _window_open(window):
                break
            if key == ord("u") and clicks:
                clicks.pop()
    finally:
        _close(window)

    pairs = [(clicks[i], clicks[i + 1]) for i in range(0, len(clicks) - 1, 2)]
    distances = [_pair_mm(to_metric, a, b, scale) for a, b in pairs]
    return distances, _annotate(image, np.array(pairs, dtype=float) / scale if pairs else [], distances)


def _open(image: np.ndarray, window: str) -> tuple[float, np.ndarray, list[tuple[int, int]]]:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_W / width, MAX_H / height)
    display = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image.copy()
    clicks: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    return scale, display, clicks


def _window_open(window: str) -> bool:
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1


def _close(window: str) -> None:
    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass


def _pair_mm(to_metric, a, b, scale: float) -> float:
    metric = to_metric(np.array([a, b], dtype=float) / scale)
    return float(np.linalg.norm(metric[1] - metric[0]))


def _mark(frame, point, label: str) -> None:
    cv2.circle(frame, point, 5, _YELLOW, -1)
    cv2.putText(frame, label, (point[0] + 8, point[1] - 8), _FONT, 0.55, _YELLOW, 1, cv2.LINE_AA)


def _banner(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 23), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)


def _segment(frame, a, b, label: str, thickness: int, font_scale: float) -> None:
    a = tuple(int(round(v)) for v in a)
    b = tuple(int(round(v)) for v in b)
    cv2.line(frame, a, b, _GREEN, thickness)
    cv2.circle(frame, a, 3 * thickness, _YELLOW, -1)
    cv2.circle(frame, b, 3 * thickness, _YELLOW, -1)
    mid = ((a[0] + b[0]) // 2 + 8, (a[1] + b[1]) // 2 - 8)
    cv2.putText(frame, label, mid, _FONT, font_scale, _GREEN, thickness, cv2.LINE_AA)


def _annotate(image: np.ndarray, pairs, distances: list[float]) -> np.ndarray:
    """Draw the measurements on a full-resolution copy for the saved report."""
    out = image.copy()
    thickness = max(2, image.shape[1] // 700)
    for (a, b), dist in zip(pairs, distances):
        _segment(out, a, b, f"{dist:.1f} mm", thickness=thickness, font_scale=0.35 * thickness)
    return out
