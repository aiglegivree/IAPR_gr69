from __future__ import annotations

import cv2
import numpy as np


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def order_box_points(points: np.ndarray) -> np.ndarray:
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        raise ValueError(f"Expected 4 box points, got shape {points.shape}")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def extract_rotated_rectangle(image_bgr: np.ndarray, box_small: np.ndarray, scale: float = 0.25) -> np.ndarray:
    ordered = order_box_points(box_small / scale)
    width = max(1, int(round(max(np.linalg.norm(ordered[2] - ordered[3]), np.linalg.norm(ordered[1] - ordered[0])))))
    height = max(1, int(round(max(np.linalg.norm(ordered[1] - ordered[2]), np.linalg.norm(ordered[0] - ordered[3])))))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image_bgr, transform, (width, height))


def paint_detected_rectangle_white(image_bgr: np.ndarray, box_small: np.ndarray, scale: float = 0.25) -> None:
    box_full = (box_small / scale).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(image_bgr, box_full, (255, 255, 255))
