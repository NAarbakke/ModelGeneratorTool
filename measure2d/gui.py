"""Interactive OpenCV click UI for measure2d.

Importing this module imports OpenCV (which needs a display) - the CLI defers
it so --help and error paths stay fast and headless-safe.

Every point is placed with a two-stage click: a coarse click on the fitted-to-
screen image, then a magnified fine click on the full-resolution pixels around
it. The fine stage is what gives real accuracy - clicking the downscaled image
directly would land on 1 screen pixel = several image pixels, the dominant error
source on high-resolution photos.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

MAX_W, MAX_H = 1400, 900  # fit the coarse viewer on screen
FINE_CROP = 160           # full-res pixels sampled around the coarse click
FINE_ZOOM = 5             # magnification of that patch (fine window = 800x800)
_GREEN = (80, 220, 80)
_YELLOW = (60, 200, 255)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_UNDO = object()    # sentinel: user pressed 'u'
_FINISH = object()  # sentinel: user pressed Enter/q (measure_loop only)


class Canceled(Exception):
    """The user pressed Esc or closed the window before finishing."""


def load_image(path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return image


# --- pure coordinate math (unit-tested; no OpenCV/display needed) -------------

def fine_crop_origin(coarse, crop: int, width: int, height: int) -> tuple[int, int]:
    """Top-left (full-res) of the FINE_CROP patch centered on the coarse click,
    clamped so the patch stays inside the image."""
    half = crop // 2
    cx = int(round(min(max(coarse[0], half), width - half)))
    cy = int(round(min(max(coarse[1], half), height - half)))
    return cx - half, cy - half


def fine_to_full(origin: tuple[int, int], fine_xy, zoom: int) -> tuple[float, float]:
    """Map a click in the magnified patch back to full-resolution coords."""
    return origin[0] + fine_xy[0] / zoom, origin[1] + fine_xy[1] / zoom


# --- interactive picking ------------------------------------------------------

def collect_points(image: np.ndarray, labels: list[str], window: str = "measure2d") -> np.ndarray:
    """Collect one point per label, in order, each via coarse+fine click.

    u = undo the last point, Esc/close = cancel. Returns an Nx2 array in
    full-resolution image pixels.
    """
    scale, display, state = _open(image, window)
    pts: list[tuple[float, float]] = []
    try:
        while len(pts) < len(labels):
            def overlay(frame, pts=pts):
                for point, label in zip(pts, labels):
                    _mark(frame, _to_disp(point, scale), label)
            res = _pick_point(
                image, display, scale, window, state,
                banner=f"point {len(pts) + 1}/{len(labels)}: {labels[len(pts)]}",
                overlay=overlay,
            )
            if res is _UNDO:
                if pts:
                    pts.pop()
            else:
                pts.append(res)
    finally:
        _close(window)
    return np.array(pts, dtype=float)


def measure_loop(
    image: np.ndarray, to_metric: Callable[[np.ndarray], np.ndarray], window: str = "measure2d"
) -> tuple[list[float], np.ndarray]:
    """Place point pairs; each completed pair is measured and printed live.

    u = undo, Enter/q/Esc/close = finish. Returns the distances (mm) and a
    full-resolution annotated copy of the image.
    """
    scale, display, state = _open(image, window)
    pts: list[tuple[float, float]] = []
    distances: list[float] = []
    printed = 0
    try:
        while True:
            def overlay(frame, pts=pts, distances=distances):
                _draw_measures(frame, pts, distances, scale)
            res = _pick_point(
                image, display, scale, window, state,
                banner=f"{len(distances)} measured - pick point pairs",
                overlay=overlay, allow_finish=True,
            )
            if res is _FINISH:
                break
            if res is _UNDO:
                if pts:
                    pts.pop()
            else:
                pts.append(res)
            distances = [_seg_mm(to_metric, pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
            printed = min(printed, len(distances))
            while printed < len(distances):
                print(f"  measurement {printed + 1}: {distances[printed]:.2f} mm")
                printed += 1
    finally:
        _close(window)

    pairs = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
    distances = [_seg_mm(to_metric, a, b) for a, b in pairs]
    annotated = _annotate(image, np.array(pairs, dtype=float) if pairs else [], distances)
    return distances, annotated


def _pick_point(image, display, scale, window, state, banner, overlay=None, allow_finish=False):
    """Acquire ONE full-resolution point: coarse aim, then magnified fine click.

    Returns (x, y) full-res floats, or the _UNDO / _FINISH sentinel; raises
    Canceled on Esc/close.
    """
    while True:
        # ---- coarse: aim on the fitted-to-screen image ----
        state["click"] = None
        while True:
            frame = display.copy()
            if overlay:
                overlay(frame)
            _crosshair(frame, state["cursor"], _YELLOW, 12)
            hint = "u=undo" + (", Enter/q=done" if allow_finish else "") + ", Esc=cancel"
            _banner(frame, f"AIM: {banner}   (click near it, {hint})")
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if not _window_open(window) or key == 27:
                raise Canceled()
            if key == ord("u"):
                return _UNDO
            if allow_finish and key in (13, ord("q")):
                return _FINISH
            if state["click"] is not None:
                coarse = (state["click"][0] / scale, state["click"][1] / scale)
                state["click"] = None
                break

        # ---- fine: click the exact pixel in the magnified patch ----
        point = _fine_pick(image, coarse, window, state)
        if point is not None:
            return point
        # right-click / 'b' in the fine view -> re-aim


def _fine_pick(image, coarse, window, state):
    """Show the full-res pixels around `coarse` magnified; return the clicked
    full-res point, or None if the user asked to re-aim."""
    height, width = image.shape[:2]
    crop = min(FINE_CROP, width, height)
    origin = fine_crop_origin(coarse, crop, width, height)
    patch = image[origin[1]:origin[1] + crop, origin[0]:origin[0] + crop]
    zoom = cv2.resize(patch, (crop * FINE_ZOOM, crop * FINE_ZOOM), interpolation=cv2.INTER_NEAREST)
    state["click"] = None
    state["rclick"] = None
    state["cursor"] = (zoom.shape[1] // 2, zoom.shape[0] // 2)
    while True:
        frame = zoom.copy()
        _crosshair(frame, state["cursor"], _GREEN, 16)
        _banner(frame, "CLICK the exact point   (right-click or b = re-aim, Esc = cancel)")
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF
        if not _window_open(window) or key == 27:
            raise Canceled()
        if key == ord("b") or state["rclick"] is not None:
            return None
        if state["click"] is not None:
            full = fine_to_full(origin, state["click"], FINE_ZOOM)
            state["click"] = None
            return full


def _open(image: np.ndarray, window: str):
    height, width = image.shape[:2]
    scale = min(1.0, MAX_W / width, MAX_H / height)
    display = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image.copy()
    state = {
        "cursor": (display.shape[1] // 2, display.shape[0] // 2),
        "click": None,
        "rclick": None,
    }

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            state["cursor"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["cursor"] = (x, y)
            state["click"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["rclick"] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    return scale, display, state


def _window_open(window: str) -> bool:
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1


def _close(window: str) -> None:
    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass


def _seg_mm(to_metric, a, b) -> float:
    metric = to_metric(np.array([a, b], dtype=float))
    return float(np.linalg.norm(metric[1] - metric[0]))


def _to_disp(point, scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def _crosshair(frame, point, color, size: int) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.line(frame, (x - size, y), (x + size, y), color, 1)
    cv2.line(frame, (x, y - size), (x, y + size), color, 1)


def _draw_measures(frame, pts, distances, scale: float) -> None:
    pairs = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
    for (a, b), dist in zip(pairs, distances):
        _segment(frame, _to_disp(a, scale), _to_disp(b, scale), f"{dist:.1f} mm", 2, 0.6)
    if len(pts) % 2:
        cv2.circle(frame, _to_disp(pts[-1], scale), 5, _YELLOW, -1)


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
