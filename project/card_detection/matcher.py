from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


def prepare_template_mask(template_mask: np.ndarray) -> np.ndarray:
    return cv2.threshold(template_mask, 127, 255, cv2.THRESH_BINARY)[1]


@lru_cache(maxsize=256)
def _rotate_template_cached(template_bytes: bytes, shape: tuple[int, int], angle_degrees: int) -> tuple[np.ndarray, np.ndarray]:
    template_mask = np.frombuffer(template_bytes, dtype=np.uint8).reshape(shape)
    height, width = template_mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_t = abs(transform[0, 0])
    sin_t = abs(transform[0, 1])
    rotated_width = int(round(height * sin_t + width * cos_t))
    rotated_height = int(round(height * cos_t + width * sin_t))
    transform[0, 2] += rotated_width / 2.0 - center[0]
    transform[1, 2] += rotated_height / 2.0 - center[1]
    rotated = cv2.warpAffine(template_mask, transform, (rotated_width, rotated_height), flags=cv2.INTER_NEAREST)
    source_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    rotated_corners = cv2.transform(source_corners.reshape(1, -1, 2), transform).reshape(-1, 2).astype(np.float32)
    return cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)[1], rotated_corners


def rotated_templates(template_mask: np.ndarray, angle_step: int) -> list[tuple[np.ndarray, np.ndarray]]:
    prepared = prepare_template_mask(template_mask)
    template_bytes = prepared.tobytes()
    shape = prepared.shape
    return [_rotate_template_cached(template_bytes, shape, angle) for angle in range(-90, 91, angle_step)]


def candidate_from_template(component_mask: np.ndarray, rotated_template: np.ndarray, rotated_corners: np.ndarray, top_left: tuple[int, int]):
    x, y = top_left
    template_height, template_width = rotated_template.shape[:2]
    if x < 0 or y < 0 or x + template_width > component_mask.shape[1] or y + template_height > component_mask.shape[0]:
        return None
    roi = component_mask[y:y + template_height, x:x + template_width]
    template_area = max(1, int(np.count_nonzero(rotated_template)))
    overlap = int(np.count_nonzero(cv2.bitwise_and(roi, rotated_template)))
    fit_ratio = overlap / template_area
    placed_corners = rotated_corners + np.array([x, y], dtype=np.float32)
    box = placed_corners.reshape(-1, 1, 2)
    bx, by, bw, bh = cv2.boundingRect(box.astype(np.int32))
    placed_template = np.zeros_like(component_mask)
    placed_template[y:y + template_height, x:x + template_width] = rotated_template
    score = overlap * fit_ratio
    return score, box, (bx, by, bw, bh), (template_width, template_height), fit_ratio, placed_template


def remove_confirmed_candidate(mask: np.ndarray, candidate) -> np.ndarray:
    remaining = mask.copy()
    box_int = candidate[1].astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(remaining, box_int, 0)
    return remaining


def outside_rectangle_ratio(labels: np.ndarray, box: np.ndarray) -> float:
    rectangle_mask = np.zeros(labels.shape, dtype=np.uint8)
    box_int = box.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(rectangle_mask, box_int, 255)
    overlapping_labels = np.unique(labels[(rectangle_mask > 0) & (labels > 0)])
    if overlapping_labels.size == 0:
        return 0.0
    blob_mask = np.isin(labels, overlapping_labels)
    total_pixels = int(np.count_nonzero(blob_mask))
    outside_pixels = int(np.count_nonzero(blob_mask & (rectangle_mask == 0)))
    return outside_pixels / total_pixels if total_pixels > 0 else 0.0


def best_template_candidate(component_mask: np.ndarray, templates: list[tuple[np.ndarray, np.ndarray]], min_fill: float, max_outside: float):
    if np.count_nonzero(component_mask) == 0:
        return None
    _, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
    best = None
    for rotated_template, rotated_corners in templates:
        th, tw = rotated_template.shape[:2]
        if th > component_mask.shape[0] or tw > component_mask.shape[1]:
            continue
        response = cv2.matchTemplate(component_mask, rotated_template, cv2.TM_CCORR)
        if response.size == 0:
            continue
        template_area = max(1, int(np.count_nonzero(rotated_template)))
        response = response / (255.0 * 255.0 * template_area)
        response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
        _, max_value, _, max_location = cv2.minMaxLoc(response)
        if max_value < min_fill:
            continue
        candidate = candidate_from_template(component_mask, rotated_template, rotated_corners, max_location)
        if candidate is None or candidate[4] < min_fill:
            continue
        if outside_rectangle_ratio(labels, candidate[1]) > max_outside:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def fit_card_mask_candidates(mask: np.ndarray, settings: dict[str, int], card_template_mask: np.ndarray, return_debug: bool = False):
    min_fill = settings["min_fill"] / 100.0
    max_outside = settings.get("max_outside", 100) / 100.0
    min_score = float(settings.get("min_score", 0))
    angle_step = max(1, int(settings.get("angle_step", 10)))
    templates = rotated_templates(card_template_mask, angle_step)
    remaining_mask = mask.copy()
    candidates = []

    while True:
        candidate = best_template_candidate(remaining_mask, templates, min_fill, max_outside)
        if candidate is None or candidate[0] < min_score:
            break
        before_pixels = int(np.count_nonzero(remaining_mask))
        candidates.append(candidate)
        remaining_mask = remove_confirmed_candidate(remaining_mask, candidate)
        after_pixels = int(np.count_nonzero(remaining_mask))
        if after_pixels >= before_pixels:
            break

    if return_debug:
        return candidates, remaining_mask, candidates.copy()
    return candidates


def detect_rectangles(mask: np.ndarray, settings: dict[str, int], card_template_mask: np.ndarray, scale: float = 0.25):
    small_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    candidates = fit_card_mask_candidates(small_mask, settings, card_template_mask)
    return small_mask, candidates
