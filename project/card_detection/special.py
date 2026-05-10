from __future__ import annotations

import cv2
import numpy as np

from .masking import build_threshold_mask, threshold_color_image, threshold_gray_image, threshold_hsv_image
from .matcher import fit_card_mask_candidates


DEFAULT_SPECIAL_SETTINGS = {
    "yellow_min_area": 40,
    "yellow_min_radius": 4,
    "yellow_min_circularity": 65,
    "yellow_min_fill": 45,
    "black_min_area": 25,
    "black_min_fill": 70,
    "black_max_long_side": 90,
    "black_min_short_side": 3,
    "black_max_aspect": 12,
}


def detect_special_shapes(
    image_bgr: np.ndarray,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    scale: float = 0.25,
    special_settings: dict[str, int] | None = None,
) -> list[dict]:
    settings = merged_special_settings(special_settings)
    yellow_mask = scaled_color_mask(image_bgr, "yellow", color_thresholds, rectangle_settings, scale)
    yellow_circle = detect_yellow_circle(yellow_mask, settings)
    if yellow_circle is not None:
        return [yellow_circle]

    black_mask = scaled_color_mask(image_bgr, "black", color_thresholds, rectangle_settings, scale)
    black_rectangle = detect_black_rectangle(black_mask, settings)
    if black_rectangle is not None:
        return [black_rectangle]

    yellow_remaining = remaining_after_card_fits(image_bgr, "yellow", color_thresholds, rectangle_settings, card_template_mask, scale)
    yellow_circle = detect_yellow_circle(yellow_remaining, settings)
    if yellow_circle is not None:
        return [yellow_circle]

    black_remaining = remaining_after_card_fits(image_bgr, "black", color_thresholds, rectangle_settings, card_template_mask, scale)
    black_rectangle = detect_black_rectangle(black_remaining, settings)
    return [] if black_rectangle is None else [black_rectangle]


def merged_special_settings(settings: dict[str, int] | None = None) -> dict[str, int]:
    merged = DEFAULT_SPECIAL_SETTINGS.copy()
    if settings:
        merged.update({key: int(value) for key, value in settings.items() if key in merged})
    return merged


def scaled_color_mask(
    image_bgr: np.ndarray,
    color: str,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    scale: float,
) -> np.ndarray:
    threshold = color_thresholds[color]
    if color == "black" and "settings" in threshold:
        settings = threshold["settings"]
        raw_mask = threshold_gray_image(image_bgr, settings["low"], settings["high"])
    else:
        raw_mask = threshold_color_image(image_bgr, threshold)
    processed_mask = build_threshold_mask(raw_mask, rectangle_settings)
    return cv2.resize(processed_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def remaining_after_card_fits(
    image_bgr: np.ndarray,
    color: str,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    small_mask = scaled_color_mask(image_bgr, color, color_thresholds, rectangle_settings, scale)
    _, remaining_mask, _ = fit_card_mask_candidates(small_mask, rectangle_settings, card_template_mask, return_debug=True)
    return remaining_mask


def detect_yellow_circle(mask: np.ndarray, settings: dict[str, int] | None = None) -> dict | None:
    settings = merged_special_settings(settings)
    min_area = float(settings["yellow_min_area"])
    min_radius = float(settings["yellow_min_radius"])
    min_circularity = settings["yellow_min_circularity"] / 100.0
    min_fill = settings["yellow_min_fill"] / 100.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or circularity < min_circularity:
            continue
        circle_area = np.pi * radius * radius
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < min_fill:
            continue
        score = area * circularity * fill
        candidate = {
            "kind": "yellow_circle",
            "color": "yellow",
            "score": score,
            "center": (float(cx), float(cy)),
            "radius": float(radius),
            "bbox": cv2.boundingRect(contour),
            "fill": float(fill),
            "circularity": float(circularity),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def detect_black_rectangle(mask: np.ndarray, settings: dict[str, int] | None = None) -> dict | None:
    settings = merged_special_settings(settings)
    min_area = float(settings["black_min_area"])
    min_fill = settings["black_min_fill"] / 100.0
    max_long_side = float(settings["black_max_long_side"])
    min_short_side = float(settings["black_min_short_side"])
    max_aspect = float(settings["black_max_aspect"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        if width <= 0 or height <= 0:
            continue
        rect_area = width * height
        fill = area / rect_area if rect_area > 0 else 0.0
        long_side = max(width, height)
        short_side = min(width, height)
        aspect = long_side / max(short_side, 1e-6)
        if fill < min_fill or long_side > max_long_side or short_side < min_short_side or aspect > max_aspect:
            continue
        score = area * fill
        candidate = {
            "kind": "black_rectangle",
            "color": "black",
            "score": score,
            "center": (float(cx), float(cy)),
            "size": (float(width), float(height)),
            "angle": float(angle),
            "bbox": cv2.boundingRect(contour),
            "fill": float(fill),
            "box": cv2.boxPoints(rect).astype(np.float32).reshape(-1, 1, 2),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best
