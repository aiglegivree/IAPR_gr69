from __future__ import annotations

import cv2
import numpy as np

from .masking import build_threshold_mask, threshold_color_image, threshold_gray_image, threshold_hsv_image
from .matcher import fit_card_mask_candidates


def detect_special_shapes(
    image_bgr: np.ndarray,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    scale: float = 0.25,
) -> list[dict]:
    yellow_mask = scaled_color_mask(image_bgr, "yellow", color_thresholds, rectangle_settings, scale)
    yellow_circle = detect_yellow_circle(yellow_mask)
    if yellow_circle is not None:
        return [yellow_circle]

    black_mask = scaled_color_mask(image_bgr, "black", color_thresholds, rectangle_settings, scale)
    black_rectangle = detect_black_rectangle(black_mask)
    if black_rectangle is not None:
        return [black_rectangle]

    yellow_remaining = remaining_after_card_fits(image_bgr, "yellow", color_thresholds, rectangle_settings, card_template_mask, scale)
    yellow_circle = detect_yellow_circle(yellow_remaining)
    if yellow_circle is not None:
        return [yellow_circle]

    black_remaining = remaining_after_card_fits(image_bgr, "black", color_thresholds, rectangle_settings, card_template_mask, scale)
    black_rectangle = detect_black_rectangle(black_remaining)
    return [] if black_rectangle is None else [black_rectangle]


def scaled_color_mask(
    image_bgr: np.ndarray,
    color: str,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    scale: float,
) -> np.ndarray:
    threshold = color_thresholds[color]
    if color == "black" and "hsv_ranges" in threshold:
        raw_mask = cv2.bitwise_or(
            threshold_hsv_image(image_bgr, threshold["hsv_ranges"]),
            threshold_gray_image(image_bgr, 0, 130),
        )
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


def detect_yellow_circle(mask: np.ndarray) -> dict | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 4 or circularity < 0.65:
            continue
        circle_area = np.pi * radius * radius
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < 0.45:
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


def detect_black_rectangle(mask: np.ndarray) -> dict | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
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
        if fill < 0.70 or long_side > 90 or short_side < 3 or aspect > 12:
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
