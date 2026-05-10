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


def paint_detected_rectangle_white(image_bgr: np.ndarray, box_small: np.ndarray, scale: float = 0.25, close_size: int = 31) -> None:
    box_full = (box_small / scale).astype(np.int32).reshape(-1, 1, 2)
    deletion_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(deletion_mask, box_full, 255)
    if close_size > 1:
        if close_size % 2 == 0:
            close_size += 1
        kernel = np.ones((close_size, close_size), dtype=np.uint8)
        deletion_mask = cv2.morphologyEx(deletion_mask, cv2.MORPH_CLOSE, kernel)
    image_bgr[deletion_mask > 0] = (255, 255, 255)


def paint_detected_mask_white(image_bgr: np.ndarray, mask_small: np.ndarray, scale: float = 0.25, make_convex: bool = True) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    deletion_mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
    deletion_mask = cv2.threshold(deletion_mask, 127, 255, cv2.THRESH_BINARY)[1]
    if make_convex:
        deletion_mask = convex_mask(deletion_mask)
    image_bgr[deletion_mask > 0] = (255, 255, 255)
    return deletion_mask


def convex_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = [contour for contour in contours if len(contour) >= 3]
    if not points:
        return mask
    hull = cv2.convexHull(np.vstack(points))
    result = np.zeros_like(mask)
    cv2.fillConvexPoly(result, hull, 255)
    return result
