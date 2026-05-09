from __future__ import annotations

import cv2
import numpy as np


def threshold_hsv_image(image_bgr: np.ndarray, ranges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def threshold_gray_image(image_bgr: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    low, high = sorted((int(low), int(high)))
    return cv2.inRange(gray, low, high)


def threshold_color_image(image_bgr: np.ndarray, threshold: dict) -> np.ndarray:
    if threshold["mode"] == "gray":
        settings = threshold["settings"]
        return threshold_gray_image(image_bgr, settings["low"], settings["high"])
    return threshold_hsv_image(image_bgr, threshold["ranges"])


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def close_mask(mask: np.ndarray, close_size: int, close_iter: int) -> np.ndarray:
    if close_size <= 0 or close_iter <= 0:
        return mask.copy()
    if close_size % 2 == 0:
        close_size += 1
    kernel = np.ones((close_size, close_size), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)


def build_threshold_mask(mask: np.ndarray, settings: dict[str, int]) -> np.ndarray:
    area_mask = filter_small_components(mask, settings["min_area"])
    return close_mask(area_mask, settings["close_size"], settings["close_iter"])
