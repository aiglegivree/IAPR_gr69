from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_COLORS, IMAGE_EXTENSIONS, CardDetectionPaths
from .special import DEFAULT_SPECIAL_SETTINGS


HSV_ARRAY_PATTERN = re.compile(r"np\.array\(\s*\[([^\]]+)\]\s*\)")
KEY_VALUE_PATTERN = re.compile(r"^\s*([^:]+)\s*:\s*(\d+)\s*$", re.MULTILINE)


def train_images(train_dir: Path) -> list[Path]:
    return [path for path in sorted(train_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]


def parse_hsv_file(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    arrays = []
    for match in HSV_ARRAY_PATTERN.finditer(path.read_text(encoding="utf-8")):
        values = [int(value.strip()) for value in match.group(1).split(",")]
        arrays.append(np.array(values, dtype=np.uint8))
    if len(arrays) % 2 != 0 or not arrays:
        raise ValueError(f"Expected lower/upper HSV pairs in {path}")
    return list(zip(arrays[0::2], arrays[1::2]))


def save_hsv_thresholds(path: Path, ranges: list[tuple[np.ndarray, np.ndarray]]) -> None:
    lines = []
    for index, (lower, upper) in enumerate(ranges, start=1):
        suffix = f"_{index}" if len(ranges) > 1 else ""
        lower_values = ", ".join(str(int(value)) for value in lower)
        upper_values = ", ".join(str(int(value)) for value in upper)
        lines.append(f"lower_hsv{suffix} = np.array([{lower_values}])")
        lines.append(f"upper_hsv{suffix} = np.array([{upper_values}])")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_gray_threshold(path: Path) -> dict[str, int]:
    defaults = {"low": 0, "high": 100}
    if not path.exists():
        return defaults
    parsed = _parse_int_settings(path)
    defaults.update({key: parsed[key] for key in defaults if key in parsed})
    return defaults


def save_gray_threshold(path: Path, settings: dict[str, int]) -> None:
    path.write_text(f"Low: {settings['low']}\nHigh: {settings['high']}\n", encoding="utf-8")


def load_color_thresholds(paths: CardDetectionPaths, colors: tuple[str, ...] = DEFAULT_COLORS) -> dict[str, dict]:
    thresholds = {}
    for color in colors:
        hsv_path = paths.hsv_file(color)
        gray_path = paths.gray_file(color)
        if hsv_path.exists():
            thresholds[color] = {"mode": "hsv", "ranges": parse_hsv_file(hsv_path)}
            if color == "black" and gray_path.exists():
                thresholds[color]["settings"] = load_gray_threshold(gray_path)
        elif color == "black" and gray_path.exists():
            thresholds[color] = {"mode": "gray", "settings": load_gray_threshold(gray_path)}
        else:
            thresholds[color] = {"mode": "hsv", "ranges": parse_hsv_file(hsv_path)}
    return thresholds


def load_rectangle_settings(path: Path) -> dict[str, int]:
    defaults = {"min_fill": 35, "min_score": 0, "angle_step": 10, "max_outside": 100, "max_candidates_per_region": 12}
    if not path.exists():
        return defaults
    parsed = _parse_int_settings(path)
    defaults.update({key: parsed[key] for key in defaults if key in parsed})
    return defaults


def load_threshold_rectangle_settings(path: Path, rectangle_defaults: dict[str, int]) -> dict[str, int]:
    defaults = {"min_area": 0, "close_size": 5, "close_iter": 1, **rectangle_defaults}
    if not path.exists():
        return defaults
    parsed = _parse_int_settings(path)
    defaults.update({key: parsed[key] for key in defaults if key in parsed})
    return defaults


def save_threshold_rectangle_settings(path: Path, settings: dict[str, int]) -> None:
    path.write_text(
        f"Min area: {settings['min_area']}\n"
        f"Close size: {settings['close_size']}\n"
        f"Close iter: {settings['close_iter']}\n"
        f"Min fill: {settings['min_fill']}\n"
        f"Min score: {settings['min_score']}\n"
        f"Angle step: {settings['angle_step']}\n"
        f"Max outside: {settings['max_outside']}\n"
        f"Max candidates per region: {settings.get('max_candidates_per_region', 12)}\n",
        encoding="utf-8",
    )


def load_special_settings(path: Path) -> dict[str, int]:
    defaults = DEFAULT_SPECIAL_SETTINGS.copy()
    if not path.exists():
        return defaults
    parsed = _parse_int_settings(path)
    defaults.update({key: parsed[key] for key in defaults if key in parsed})
    return defaults


def save_special_settings(path: Path, settings: dict[str, int]) -> None:
    merged = DEFAULT_SPECIAL_SETTINGS.copy()
    merged.update({key: int(value) for key, value in settings.items() if key in merged})
    path.write_text(
        f"Yellow min radius: {merged['yellow_min_radius']}\n"
        f"Yellow max radius: {merged['yellow_max_radius']}\n"
        f"Yellow min circularity: {merged['yellow_min_circularity']}\n"
        f"Yellow min fill: {merged['yellow_min_fill']}\n"
        f"Black min area: {merged['black_min_area']}\n"
        f"Black min fill: {merged['black_min_fill']}\n"
        f"Black max long side: {merged['black_max_long_side']}\n"
        f"Black min short side: {merged['black_min_short_side']}\n"
        f"Black max aspect: {merged['black_max_aspect']}\n",
        encoding="utf-8",
    )


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image_bgr, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_card_template(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load card template mask: {path}. Create it with card_mask.ipynb first.")
    return mask


def _parse_int_settings(path: Path) -> dict[str, int]:
    return {
        key.strip().lower().replace(" ", "_"): int(value)
        for key, value in KEY_VALUE_PATTERN.findall(path.read_text(encoding="utf-8"))
    }
