from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import bbox_iou, extract_rotated_rectangle, paint_detected_rectangle_white
from .masking import build_threshold_mask, threshold_color_image
from .matcher import fit_card_mask_candidates
from .special import detect_special_shapes


@dataclass
class CardDetectionResult:
    cards: list[tuple[str, np.ndarray]]
    overlay: np.ndarray
    debug_views: list[tuple[str, np.ndarray]]
    rows: list[dict]


def detect_cards_in_image(
    image_bgr: np.ndarray,
    image_rgb: np.ndarray,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    colors: tuple[str, ...],
    scale: float = 0.25,
    iou_threshold: float = 0.50,
    capture_debug: bool = True,
    iterative: bool = False,
    detect_special: bool = True,
) -> CardDetectionResult:
    cleaned_image_bgr = image_bgr.copy()
    overlay = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()
    debug_views: list[tuple[str, np.ndarray]] = []
    cards: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []
    rank = 1

    candidates = collect_candidates(
        cleaned_image_bgr,
        color_thresholds,
        rectangle_settings,
        card_template_mask,
        colors,
        scale,
        iou_threshold,
        debug_views if capture_debug else None,
    )
    while candidates:
        color, score, box, bbox, _, fill_ratio, _ = candidates.pop(0)
        box_int = box.astype(np.int32).reshape(-1, 1, 2)
        x, y, _, _ = cv2.boundingRect(box_int)
        cv2.polylines(overlay, [box_int], True, (255, 0, 0), 2)
        cv2.putText(overlay, f"#{rank} {color}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
        try:
            crop = extract_rotated_rectangle(cleaned_image_bgr, box, scale=scale)
        except ValueError:
            paint_detected_rectangle_white(cleaned_image_bgr, box, scale=scale)
            candidates = collect_candidates(cleaned_image_bgr, color_thresholds, rectangle_settings, card_template_mask, colors, scale, iou_threshold, None)
            continue
        title = f"#{rank} {color} score={score:.1f} fill={fill_ratio:.2f}"
        cards.append((title, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        rows.append({"rank": rank, "color": color, "score": score, "fill": fill_ratio, "bbox": bbox})
        paint_detected_rectangle_white(cleaned_image_bgr, box, scale=scale)
        rank += 1
        if iterative:
            candidates = collect_candidates(cleaned_image_bgr, color_thresholds, rectangle_settings, card_template_mask, colors, scale, iou_threshold, None)

    if detect_special:
        for special in detect_special_shapes(image_bgr, color_thresholds, rectangle_settings, card_template_mask, scale=scale):
            draw_special_detection(overlay, special)
            special_rank = len(rows) + 1
            row = {"rank": special_rank, **special}
            rows.append(row)

    return CardDetectionResult(cards=cards, overlay=overlay, debug_views=debug_views, rows=rows)


def collect_candidates(
    frame_bgr: np.ndarray,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    colors: tuple[str, ...],
    scale: float,
    iou_threshold: float,
    debug_views: list[tuple[str, np.ndarray]] | None,
):
    all_candidates = []
    for color in colors:
        raw_mask = threshold_color_image(frame_bgr, color_thresholds[color])
        processed_mask = build_threshold_mask(raw_mask, rectangle_settings)
        small_mask = cv2.resize(processed_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if debug_views is None:
            candidates = fit_card_mask_candidates(small_mask, rectangle_settings, card_template_mask)
            remaining_mask = None
        else:
            candidates, remaining_mask, _ = fit_card_mask_candidates(small_mask, rectangle_settings, card_template_mask, return_debug=True)
        if debug_views is not None:
            debug_views.append((f"{color} threshold", raw_mask))
            debug_views.append((f"{color} area+close", small_mask))
            debug_views.append((f"{color} remaining after fit", remaining_mask))
        unique_color_candidates = []
        for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(bbox_iou(candidate[2], kept[2]) < iou_threshold for kept in unique_color_candidates):
                unique_color_candidates.append(candidate)
        for candidate in unique_color_candidates:
            all_candidates.append((color, *candidate))

    return all_candidates


def draw_special_detection(overlay: np.ndarray, special: dict) -> None:
    if special["kind"] == "yellow_circle":
        cx, cy = special["center"]
        radius = special["radius"]
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(overlay, center, int(round(radius)), (255, 255, 0), 2)
        cv2.putText(overlay, "yellow circle", (center[0] + 6, max(20, center[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        return

    if special["kind"] == "black_rectangle":
        box = special["box"].astype(np.int32).reshape(-1, 1, 2)
        x, y, _, _ = cv2.boundingRect(box)
        cv2.polylines(overlay, [box], True, (0, 0, 0), 2)
        cv2.putText(overlay, "black rectangle", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
