#!/usr/bin/env python3
"""
Tune Canny edge detection on images in thresholded_refs, one image at a time.

Usage:
    python3 canny.py

Keys:
    q or Esc  Quit
    n or Right Next image
    b or Left  Previous image
    p         Print current Canny settings
    s         Save current Canny settings to canny_values.txt
    w         Write Canny edge images to canny_refs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Canny Tuner"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively tune Canny on all thresholded reference images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("thresholded_refs"),
        help="Folder containing thresholded reference images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("canny_refs"),
        help="Folder where edge images are written when pressing w.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="Resize preview to this maximum width while preserving aspect ratio.",
    )
    return parser.parse_args()


def nothing(_: int) -> None:
    pass


def create_trackbars() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold 1", WINDOW_NAME, 50, 500, nothing)
    cv2.createTrackbar("Threshold 2", WINDOW_NAME, 150, 500, nothing)
    cv2.createTrackbar("Blur", WINDOW_NAME, 0, 20, nothing)
    cv2.createTrackbar("Aperture", WINDOW_NAME, 1, 2, nothing)
    cv2.createTrackbar("L2 gradient", WINDOW_NAME, 0, 1, nothing)


def current_settings() -> tuple[int, int, int, int, bool]:
    threshold_1 = cv2.getTrackbarPos("Threshold 1", WINDOW_NAME)
    threshold_2 = cv2.getTrackbarPos("Threshold 2", WINDOW_NAME)
    blur = cv2.getTrackbarPos("Blur", WINDOW_NAME)
    aperture = 3 + 2 * cv2.getTrackbarPos("Aperture", WINDOW_NAME)
    l2_gradient = cv2.getTrackbarPos("L2 gradient", WINDOW_NAME) == 1
    return threshold_1, threshold_2, blur, aperture, l2_gradient


def format_settings(
    threshold_1: int,
    threshold_2: int,
    blur: int,
    aperture: int,
    l2_gradient: bool,
) -> str:
    return (
        f"threshold_1 = {threshold_1}\n"
        f"threshold_2 = {threshold_2}\n"
        f"blur = {blur}\n"
        f"aperture_size = {aperture}\n"
        f"L2gradient = {l2_gradient}"
    )


def image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {input_dir}")

    paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    return paths


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def apply_canny(
    image: np.ndarray,
    threshold_1: int,
    threshold_2: int,
    blur: int,
    aperture: int,
    l2_gradient: bool,
) -> np.ndarray:
    if blur > 0:
        kernel_size = 2 * blur + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return cv2.Canny(
        image,
        threshold_1,
        threshold_2,
        apertureSize=aperture,
        L2gradient=l2_gradient,
    )


def resize_for_preview(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image

    scale = max_width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)


def make_preview(
    edges: np.ndarray,
    image_name: str,
    image_index: int,
    image_count: int,
    max_width: int,
) -> np.ndarray:
    preview = resize_for_preview(edges, max_width)
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    label = f"{image_index + 1}/{image_count}  {image_name}"
    label_bar = np.zeros((32, preview.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        label_bar,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([label_bar, preview])


def write_edges(
    paths: list[Path],
    output_dir: Path,
    threshold_1: int,
    threshold_2: int,
    blur: int,
    aperture: int,
    l2_gradient: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for path in paths:
        image = load_image(path)
        edges = apply_canny(image, threshold_1, threshold_2, blur, aperture, l2_gradient)
        output_path = output_dir / f"{path.stem}_canny.png"
        if not cv2.imwrite(str(output_path), edges):
            raise ValueError(f"Could not write image: {output_path}")
        saved_count += 1

    return saved_count


def main() -> int:
    args = parse_args()
    paths = image_paths(args.input_dir)
    image_index = 0
    image = load_image(paths[image_index])

    create_trackbars()
    print(
        f"Found {len(paths)} images in {args.input_dir}. "
        "Press n/right for next, b/left for previous, p to print settings, "
        "s to save settings, w to write all edges, q to quit."
    )

    while True:
        threshold_1, threshold_2, blur, aperture, l2_gradient = current_settings()
        edges = apply_canny(
            image,
            threshold_1,
            threshold_2,
            blur,
            aperture,
            l2_gradient,
        )
        preview = make_preview(
            edges,
            paths[image_index].stem,
            image_index,
            len(paths),
            args.max_width,
        )
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 83):
            image_index = (image_index + 1) % len(paths)
            image = load_image(paths[image_index])
            print(f"Showing {image_index + 1}/{len(paths)}: {paths[image_index].name}")
        elif key in (ord("b"), 81):
            image_index = (image_index - 1) % len(paths)
            image = load_image(paths[image_index])
            print(f"Showing {image_index + 1}/{len(paths)}: {paths[image_index].name}")
        elif key == ord("p"):
            print(format_settings(threshold_1, threshold_2, blur, aperture, l2_gradient))
        elif key == ord("s"):
            output_path = Path("canny_values.txt")
            output_path.write_text(
                format_settings(threshold_1, threshold_2, blur, aperture, l2_gradient)
                + "\n",
                encoding="utf-8",
            )
            print(f"Saved Canny settings to {output_path.resolve()}")
        elif key == ord("w"):
            saved_count = write_edges(
                paths,
                args.output_dir,
                threshold_1,
                threshold_2,
                blur,
                aperture,
                l2_gradient,
            )
            print(f"Saved {saved_count} Canny edge images to {args.output_dir.resolve()}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
