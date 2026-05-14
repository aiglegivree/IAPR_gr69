#!/usr/bin/env python3
"""
Apply saved HSV thresholds to the reference images and selected extra images.

Reads:
    black_hsv.txt
    blue_hsv.txt
    yellow_hsv.txt
    green_hsv.txt
    red_hsv.txt

and writes one binary threshold mask per color/image pair to:
    thresholded_refs/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


COLORS = ("black", "blue", "yellow", "green", "red")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HSV_ARRAY_PATTERN = re.compile(r"np\.array\(\s*\[([^\]]+)\]\s*\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold reference images with saved HSV ranges."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("iapr-26-uno-vision-challenge/reference_images"),
        help="Directory containing the reference images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thresholded_refs"),
        help="Directory where thresholded images will be saved.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=15000,
        help="Remove connected foreground objects smaller than this area.",
    )
    parser.add_argument(
        "--extra-image",
        action="append",
        type=Path,
        default=[Path("iapr-26-uno-vision-challenge/test_images/L1000997.jpg")],
        help="Additional image to threshold. Can be passed multiple times.",
    )
    return parser.parse_args()


def parse_hsv_file(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    text = path.read_text(encoding="utf-8")
    arrays = []

    for match in HSV_ARRAY_PATTERN.finditer(text):
        values = [int(value.strip()) for value in match.group(1).split(",")]
        if len(values) != 3:
            raise ValueError(f"Expected 3 HSV values in {path}, got {values}")
        arrays.append(np.array(values, dtype=np.uint8))

    if len(arrays) % 2 != 0 or not arrays:
        raise ValueError(f"Expected lower/upper HSV pairs in {path}")

    return list(zip(arrays[0::2], arrays[1::2]))


def load_thresholds() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    thresholds = {}
    for color in COLORS:
        path = Path(f"{color}_hsv.txt")
        if not path.exists():
            raise FileNotFoundError(f"Missing HSV file: {path}")
        thresholds[color] = parse_hsv_file(path)
    return thresholds


def threshold_image(
    image: np.ndarray,
    ranges: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    return combined_mask


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def reference_images(reference_dir: Path) -> list[Path]:
    if not reference_dir.exists():
        raise FileNotFoundError(f"Missing reference image directory: {reference_dir}")

    images = [
        path
        for path in sorted(reference_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No reference images found in {reference_dir}")

    return images


def threshold_input_images(reference_dir: Path, extra_images: list[Path]) -> list[Path]:
    images = reference_images(reference_dir)

    for image_path in extra_images:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing extra image: {image_path}")
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported extra image extension: {image_path}")
        images.append(image_path)

    return images


def main() -> int:
    args = parse_args()
    thresholds = load_thresholds()
    images = threshold_input_images(args.reference_dir, args.extra_image)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        for color, ranges in thresholds.items():
            mask = threshold_image(image, ranges)
            mask = remove_small_objects(mask, args.min_area)
            output_path = args.output_dir / f"{image_path.stem}_{color}.png"
            if not cv2.imwrite(str(output_path), mask):
                raise ValueError(f"Could not write image: {output_path}")
            saved_count += 1

    print(
        f"Saved {saved_count} thresholded images to {args.output_dir} "
        f"with min area {args.min_area}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
