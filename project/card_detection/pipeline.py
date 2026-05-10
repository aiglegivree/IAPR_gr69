from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import bbox_iou, extract_rotated_rectangle, paint_detected_mask_white
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
    removal_make_convex: bool = True,
    show_progress: bool = False,
) -> CardDetectionResult:
    progress = DetectionProgress(show_progress)
    cleaned_image_bgr = image_bgr.copy()
    overlay = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()
    debug_views: list[tuple[str, np.ndarray]] = []
    cards: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []
    rank = 1
    matching_scale = 0.25

    if detect_special:
        progress.step("Detecting specials")
        for special in detect_special_shapes(image_bgr, color_thresholds, rectangle_settings, card_template_mask, scale=scale):
            draw_special_detection(overlay, special)
            rows.append({"rank": rank, **special})
            paint_special_detection_white(cleaned_image_bgr, special, scale=scale)
            rank += 1

    card_colors = black_first(colors)
    last_removal_mask = None
    overlay_factor = scale / matching_scale

    for color_name in card_colors:
        progress.step(f"Collecting {color_name} candidates")
        candidates = collect_candidates(
            cleaned_image_bgr,
            color_thresholds,
            rectangle_settings,
            card_template_mask,
            (color_name,),
            matching_scale,
            iou_threshold,
            debug_views if capture_debug else None,
            progress=progress,
            label=f"{color_name} candidates",
        )
        while candidates:
            color, score, box, bbox, _, fill_ratio, detected_card_mask, outside_ratio = candidates.pop(0)
            progress.step(f"Extracting #{rank} {color}")
            row = handle_card_candidate(
                cleaned_image_bgr,
                overlay,
                cards,
                color_thresholds,
                color,
                score,
                box,
                bbox,
                fill_ratio,
                outside_ratio,
                detected_card_mask,
                rank,
                matching_scale,
                overlay_factor,
                removal_make_convex,
            )
            if row is not None:
                rows.append(row)
                last_removal_mask = row.pop("_removal_mask")
                rank += 1
            else:
                last_removal_mask = paint_detected_mask_white(cleaned_image_bgr, detected_card_mask, scale=matching_scale, make_convex=removal_make_convex)
        if color_name == "black" and capture_debug:
            append_black_removal_debug(debug_views, cleaned_image_bgr, last_removal_mask, scale)

    return CardDetectionResult(cards=cards, overlay=overlay, debug_views=debug_views, rows=rows)


class DetectionProgress:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._tqdm = None
        if enabled:
            try:
                from tqdm.auto import tqdm

                self._tqdm = tqdm
            except ImportError:
                self._tqdm = None

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        if self._tqdm is None:
            print(message)
        else:
            self._tqdm.write(message)

    def iter_colors(self, colors: tuple[str, ...], label: str):
        if not self.enabled or self._tqdm is None:
            return colors
        return self._tqdm(colors, desc=label, leave=False)


def handle_card_candidate(
    cleaned_image_bgr: np.ndarray,
    overlay: np.ndarray,
    cards: list[tuple[str, np.ndarray]],
    color_thresholds: dict[str, dict],
    color: str,
    score: float,
    box: np.ndarray,
    bbox: tuple[int, int, int, int],
    fill_ratio: float,
    outside_ratio: float,
    detected_card_mask: np.ndarray,
    rank: int,
    matching_scale: float,
    overlay_factor: float,
    removal_make_convex: bool,
) -> dict | None:
    box_overlay = (box * overlay_factor).astype(np.int32).reshape(-1, 1, 2)
    x, y, _, _ = cv2.boundingRect(box_overlay)
    cv2.polylines(overlay, [box_overlay], True, (255, 0, 0), 2)
    cv2.putText(overlay, f"#{rank} {color}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    try:
        crop = extract_rotated_rectangle(cleaned_image_bgr, box, scale=matching_scale)
    except ValueError:
        return None

    crop = keep_color_in_crop(crop, color_thresholds[color])
    title = f"#{rank} {color} score={score:.1f} fill={fill_ratio:.2f}"
    cards.append((title, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    removal_mask = paint_detected_mask_white(cleaned_image_bgr, detected_card_mask, scale=matching_scale, make_convex=removal_make_convex)
    return {
        "rank": rank,
        "color": color,
        "score": score,
        "fill": fill_ratio,
        "outside": outside_ratio,
        "bbox": bbox,
        "_removal_mask": removal_mask,
    }


def black_first(colors: tuple[str, ...]) -> tuple[str, ...]:
    if "black" not in colors:
        return colors
    return ("black", *(color for color in colors if color != "black"))


def append_black_removal_debug(debug_views: list[tuple[str, np.ndarray]], image_bgr: np.ndarray, removal_mask: np.ndarray | None, scale: float) -> None:
    if removal_mask is not None:
        debug_mask = cv2.resize(removal_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        debug_views.append(("black removal convex mask", debug_mask))
    debug_image = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    debug_views.append(("after black removal", cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)))


def collect_candidates(
    frame_bgr: np.ndarray,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    colors: tuple[str, ...],
    scale: float,
    iou_threshold: float,
    debug_views: list[tuple[str, np.ndarray]] | None,
    progress: DetectionProgress | None = None,
    label: str = "candidates",
):
    all_candidates = []
    color_iter = progress.iter_colors(colors, label) if progress is not None else colors
    for color in color_iter:
        if progress is not None:
            progress.step(f"{label}: threshold/cleanup {color}")
        raw_mask = threshold_color_image(frame_bgr, color_thresholds[color])
        processed_mask = build_threshold_mask(raw_mask, rectangle_settings)
        small_mask = cv2.resize(processed_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if progress is not None:
            progress.step(f"{label}: template matching {color}")
        if debug_views is None:
            candidates = fit_card_mask_candidates(small_mask, rectangle_settings, card_template_mask)
            remaining_mask = None
        else:
            candidates, remaining_mask, _ = fit_card_mask_candidates(
                small_mask,
                rectangle_settings,
                card_template_mask,
                return_debug=True,
            )
        if debug_views is not None:
            debug_views.append((f"{color} threshold", resize_debug_view(raw_mask, scale)))
            debug_views.append((f"{color} area+close", small_mask))
            debug_views.append((f"{color} remaining after fit", remaining_mask))
        unique_color_candidates = []
        for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(bbox_iou(candidate[2], kept[2]) < iou_threshold for kept in unique_color_candidates):
                unique_color_candidates.append(candidate_with_detected_mask(candidate, small_mask))
        for candidate in unique_color_candidates:
            all_candidates.append((color, *candidate))

    return all_candidates


def candidate_with_detected_mask(candidate, color_mask: np.ndarray):
    score, box, bbox, template_size, fill_ratio, placed_template, outside_ratio = candidate
    detected_mask = cv2.bitwise_and(color_mask, placed_template)
    return score, box, bbox, template_size, fill_ratio, detected_mask, outside_ratio


def resize_debug_view(image: np.ndarray | None, scale: float) -> np.ndarray | None:
    if image is None or scale == 1.0:
        return image
    interpolation = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def keep_color_in_crop(crop_bgr: np.ndarray, color_threshold: dict) -> np.ndarray:
    mask = threshold_color_image(crop_bgr, color_threshold)
    filtered = np.full_like(crop_bgr, 255)
    filtered[mask > 0] = crop_bgr[mask > 0]
    return filtered


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


def paint_special_detection_white(image_bgr: np.ndarray, special: dict, scale: float = 0.25) -> None:
    if special["kind"] == "yellow_circle":
        cx, cy = special["center"]
        center = (int(round(cx / scale)), int(round(cy / scale)))
        radius = int(round(special["radius"] / scale))
        cv2.circle(image_bgr, center, radius, (255, 255, 255), -1)
        return

    if special["kind"] == "black_rectangle":
        box_full = (special["box"] / scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(image_bgr, box_full, (255, 255, 255))
