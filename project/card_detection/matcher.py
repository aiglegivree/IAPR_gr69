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


def rotated_templates(template_mask: np.ndarray, angle_step: int) -> list[tuple[np.ndarray, np.ndarray, int, int, int]]:
    prepared = prepare_template_mask(template_mask)
    template_bytes = prepared.tobytes()
    shape = prepared.shape
    templates = []
    for angle in range(-90, 91, angle_step):
        rotated_template, rotated_corners = _rotate_template_cached(template_bytes, shape, angle)
        template_height, template_width = rotated_template.shape[:2]
        template_area = max(1, int(np.count_nonzero(rotated_template)))
        templates.append((rotated_template, rotated_corners, template_area, template_height, template_width))
    return templates


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


def outside_template_details(labels: np.ndarray, placed_template: np.ndarray, tolerance: int = 2):
    template_mask = cv2.threshold(placed_template, 127, 255, cv2.THRESH_BINARY)[1]
    if tolerance > 0:
        kernel_size = 2 * tolerance + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        template_mask = cv2.dilate(template_mask, kernel)
    overlapping_labels = np.unique(labels[(template_mask > 0) & (labels > 0)])
    if overlapping_labels.size == 0:
        empty = np.zeros(labels.shape, dtype=bool)
        return 0.0, empty, template_mask, empty
    blob_mask = np.isin(labels, overlapping_labels)
    total_pixels = int(np.count_nonzero(blob_mask))
    outside_mask = blob_mask & (template_mask == 0)
    outside_pixels = int(np.count_nonzero(outside_mask))
    ratio = outside_pixels / total_pixels if total_pixels > 0 else 0.0
    return ratio, blob_mask, template_mask, outside_mask


def outside_template_ratio(labels: np.ndarray, placed_template: np.ndarray, tolerance: int = 2) -> float:
    ratio, _, _, _ = outside_template_details(labels, placed_template, tolerance)
    return ratio


def local_outside_details(component_mask: np.ndarray, inside_mask: np.ndarray, candidate_mask: np.ndarray):
    local_mask = cv2.bitwise_and(component_mask, candidate_mask) > 0
    inside = inside_mask > 0
    total_pixels = int(np.count_nonzero(local_mask))
    outside_mask = local_mask & ~inside
    outside_pixels = int(np.count_nonzero(outside_mask))
    ratio = outside_pixels / total_pixels if total_pixels > 0 else 0.0
    return ratio, local_mask, inside_mask, outside_mask


def candidate_rectangle_mask(shape: tuple[int, int], box: np.ndarray) -> np.ndarray:
    rectangle_mask = np.zeros(shape, dtype=np.uint8)
    box_int = box.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(rectangle_mask, box_int, 255)
    return rectangle_mask


def outside_rectangle_ratio(labels: np.ndarray, box: np.ndarray) -> float:
    rectangle_mask = candidate_rectangle_mask(labels.shape, box)
    return outside_template_ratio(labels, rectangle_mask, tolerance=0)


def outside_debug_image(component_mask: np.ndarray, blob_mask: np.ndarray, template_mask: np.ndarray, outside_mask: np.ndarray) -> np.ndarray:
    debug = np.repeat(component_mask[:, :, None], 3, axis=2)
    debug[blob_mask] = (80, 80, 255)
    debug[template_mask > 0] = (0, 220, 0)
    debug[outside_mask] = (255, 0, 0)
    return debug


def best_template_candidate(
    component_mask: np.ndarray,
    templates: list[tuple[np.ndarray, np.ndarray, int, int, int]],
    min_fill: float,
    max_candidates_per_region: int,
    outside_debug: list[tuple[str, np.ndarray]] | None = None,
):
    if np.count_nonzero(component_mask) == 0:
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
    best = None
    search_regions = component_search_regions(stats, component_mask.shape)
    for rotated_template, rotated_corners, template_area, th, tw in templates:
        if th > component_mask.shape[0] or tw > component_mask.shape[1]:
            continue
        for x0, y0, x1, y1 in search_regions:
            roi = component_mask[y0:y1, x0:x1]
            if th > roi.shape[0] or tw > roi.shape[1]:
                continue
            response = cv2.matchTemplate(roi, rotated_template, cv2.TM_CCORR)
            if response.size == 0:
                continue
            response = response / (255.0 * 255.0 * template_area)
            response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
            for local_location in top_response_locations(response, min_fill, max_candidates=max_candidates_per_region, suppression_radius=max(th, tw) // 4):
                max_location = (local_location[0] + x0, local_location[1] + y0)
                candidate = candidate_from_template(component_mask, rotated_template, rotated_corners, max_location)
                if candidate is None or candidate[4] < min_fill:
                    continue
                outside_ratio = outside_rectangle_ratio(labels, candidate[1])
                candidate = (*candidate, outside_ratio)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best


def component_search_regions(stats: np.ndarray, image_shape: tuple[int, int], min_pixels: int = 25) -> list[tuple[int, int, int, int]]:
    height, width = image_shape
    regions = []
    for label in range(1, stats.shape[0]):
        x, y, w, h, area = stats[label]
        if area < min_pixels:
            continue
        pad = max(w, h)
        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(width, int(x + w + pad))
        y1 = min(height, int(y + h + pad))
        regions.append((x0, y0, x1, y1))
    return merge_search_regions(regions)


def merge_search_regions(regions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for region in sorted(regions, key=lambda item: (item[1], item[0])):
        rx0, ry0, rx1, ry1 = region
        for index, (x0, y0, x1, y1) in enumerate(merged):
            if rx0 <= x1 and rx1 >= x0 and ry0 <= y1 and ry1 >= y0:
                merged[index] = (min(x0, rx0), min(y0, ry0), max(x1, rx1), max(y1, ry1))
                break
        else:
            merged.append(region)
    return merged


def top_response_locations(response: np.ndarray, min_value: float, max_candidates: int, suppression_radius: int):
    remaining = response.copy()
    radius = max(1, int(suppression_radius))
    for _ in range(max_candidates):
        _, max_value, _, max_location = cv2.minMaxLoc(remaining)
        if max_value < min_value:
            break
        yield max_location
        x, y = max_location
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(remaining.shape[1], x + radius + 1)
        y1 = min(remaining.shape[0], y + radius + 1)
        remaining[y0:y1, x0:x1] = 0.0


def fit_card_mask_candidates(
    mask: np.ndarray,
    settings: dict[str, int],
    card_template_mask: np.ndarray,
    return_debug: bool = False,
    outside_debug: list[tuple[str, np.ndarray]] | None = None,
):
    min_fill = settings["min_fill"] / 100.0
    max_outside = settings.get("max_outside", 100) / 100.0
    min_score = float(settings.get("min_score", 0))
    angle_step = max(1, int(settings.get("angle_step", 10)))
    max_candidates_per_region = max(1, int(settings.get("max_candidates_per_region", 12)))
    templates = rotated_templates(card_template_mask, angle_step)
    remaining_mask = mask.copy()
    candidates = []

    while True:
        candidate = best_template_candidate(remaining_mask, templates, min_fill, max_candidates_per_region, outside_debug)
        if candidate is None or candidate[0] < min_score:
            break
        before_pixels = int(np.count_nonzero(remaining_mask))
        candidates.append(candidate)
        remaining_mask = remove_confirmed_candidate(remaining_mask, candidate)
        after_pixels = int(np.count_nonzero(remaining_mask))
        if after_pixels >= before_pixels:
            break
    accepted_candidates = [candidate for candidate in candidates if candidate[6] <= max_outside]
    accepted_remaining_mask = remove_candidates(mask, accepted_candidates)

    if return_debug:
        return accepted_candidates, accepted_remaining_mask, candidates.copy()
    return accepted_candidates


def remove_candidates(mask: np.ndarray, candidates: list) -> np.ndarray:
    remaining = mask.copy()
    for candidate in candidates:
        remaining = remove_confirmed_candidate(remaining, candidate)
    return remaining


def detect_rectangles(mask: np.ndarray, settings: dict[str, int], card_template_mask: np.ndarray, scale: float = 0.25):
    small_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    candidates = fit_card_mask_candidates(small_mask, settings, card_template_mask)
    return small_mask, candidates
