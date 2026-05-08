#!/usr/bin/env python3
"""
Interactive HSV threshold tuner.

Usage:
    python hsv_tuner path/to/image.jpg

Keys:
    q or Esc  Quit
    p         Print current HSV values
    s         Save current HSV values to hsv_values.txt
    m         Toggle between masked image and binary mask preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "HSV Tuner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HSV lower/upper bounds for an input image."
    )
    parser.add_argument("image_path", help="Path to the image to tune.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1000,
        help="Resize preview to this maximum width while preserving aspect ratio.",
    )
    return parser.parse_args()


def resize_for_preview(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image

    scale = max_width / image.shape[1]
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)


def nothing(_: int) -> None:
    pass


def create_trackbars() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H1 min", WINDOW_NAME, 0, 179, nothing)
    cv2.createTrackbar("H1 max", WINDOW_NAME, 179, 179, nothing)
    cv2.createTrackbar("Use H2", WINDOW_NAME, 0, 1, nothing)
    cv2.createTrackbar("H2 min", WINDOW_NAME, 170, 179, nothing)
    cv2.createTrackbar("H2 max", WINDOW_NAME, 179, 179, nothing)
    cv2.createTrackbar("S min", WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar("S max", WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar("V min", WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar("V max", WINDOW_NAME, 255, 255, nothing)


def current_bounds() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    s_min = cv2.getTrackbarPos("S min", WINDOW_NAME)
    s_max = cv2.getTrackbarPos("S max", WINDOW_NAME)
    v_min = cv2.getTrackbarPos("V min", WINDOW_NAME)
    v_max = cv2.getTrackbarPos("V max", WINDOW_NAME)

    lower_1 = np.array(
        [
            cv2.getTrackbarPos("H1 min", WINDOW_NAME),
            s_min,
            v_min,
        ]
    )
    upper_1 = np.array(
        [
            cv2.getTrackbarPos("H1 max", WINDOW_NAME),
            s_max,
            v_max,
        ]
    )
    lower_2 = np.array(
        [
            cv2.getTrackbarPos("H2 min", WINDOW_NAME),
            s_min,
            v_min,
        ]
    )
    upper_2 = np.array(
        [
            cv2.getTrackbarPos("H2 max", WINDOW_NAME),
            s_max,
            v_max,
        ]
    )
    use_h2 = cv2.getTrackbarPos("Use H2", WINDOW_NAME) == 1
    return lower_1, upper_1, lower_2, upper_2, use_h2


def make_mask(
    hsv: np.ndarray,
    lower_1: np.ndarray,
    upper_1: np.ndarray,
    lower_2: np.ndarray,
    upper_2: np.ndarray,
    use_h2: bool,
) -> np.ndarray:
    mask = cv2.inRange(hsv, lower_1, upper_1)
    if use_h2:
        mask_2 = cv2.inRange(hsv, lower_2, upper_2)
        mask = cv2.bitwise_or(mask, mask_2)
    return mask


def format_bounds(
    lower_1: np.ndarray,
    upper_1: np.ndarray,
    lower_2: np.ndarray,
    upper_2: np.ndarray,
    use_h2: bool,
) -> str:
    lines = [
        f"lower_hsv_1 = np.array({lower_1.tolist()})",
        f"upper_hsv_1 = np.array({upper_1.tolist()})",
    ]
    if use_h2:
        lines.extend(
            [
                f"lower_hsv_2 = np.array({lower_2.tolist()})",
                f"upper_hsv_2 = np.array({upper_2.tolist()})",
                "mask = cv2.bitwise_or(",
                "    cv2.inRange(hsv, lower_hsv_1, upper_hsv_1),",
                "    cv2.inRange(hsv, lower_hsv_2, upper_hsv_2),",
                ")",
            ]
        )
    else:
        lines.append("mask = cv2.inRange(hsv, lower_hsv_1, upper_hsv_1)")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path).expanduser()

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    image = resize_for_preview(image, args.max_width)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    create_trackbars()
    show_mask = False

    print(
        "Adjust sliders. Turn Use H2 on for wrap-around colors like red. "
        "Press p to print values, s to save, m to toggle mask, q to quit."
    )

    while True:
        lower_1, upper_1, lower_2, upper_2, use_h2 = current_bounds()
        mask = make_mask(hsv, lower_1, upper_1, lower_2, upper_2, use_h2)
        result = cv2.bitwise_and(image, image, mask=mask)

        if show_mask:
            preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            preview = result

        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
        elif key == ord("p"):
            print(format_bounds(lower_1, upper_1, lower_2, upper_2, use_h2))
        elif key == ord("s"):
            output_path = Path("hsv_values.txt")
            output_path.write_text(
                format_bounds(lower_1, upper_1, lower_2, upper_2, use_h2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved HSV values to {output_path.resolve()}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
