from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent


def find_train_dir():
    candidates = [
        BASE_DIR / "iapr-26-uno-vision-challenge" / "train_images",
        BASE_DIR / "project" / "iapr-26-uno-vision-challenge" / "train_images",
        BASE_DIR.parent / "project" / "iapr-26-uno-vision-challenge" / "train_images",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find train_images directory. Searched:\n{searched}")


TRAIN_DIR = find_train_dir()

IMAGE_TYPES = {
    "L1000770-L1000857": (1000770, 1000857),
    "L1000902-L1000983": (1000902, 1000983),
}


def image_mean_gray(image_path):
    image = Image.open(image_path).convert("L")
    return float(np.array(image, dtype=np.float32).mean())


def image_paths_for_range(start, end):
    return [
        TRAIN_DIR / f"L{image_id}.jpg"
        for image_id in range(start, end + 1)
    ]


def summarize_image_type(label, start, end):
    image_paths = image_paths_for_range(start, end)
    existing_paths = [path for path in image_paths if path.exists()]
    missing_paths = [path for path in image_paths if not path.exists()]

    if not existing_paths:
        raise FileNotFoundError(f"No images found for {label}")

    gray_means = np.array(
        [image_mean_gray(path) for path in existing_paths],
        dtype=np.float32,
    )

    print(label)
    print(f"  Images: {len(gray_means)}")
    if missing_paths:
        print(f"  Missing images in range: {len(missing_paths)}")
    print(f"  Mean grayscale: {gray_means.mean():.2f}")
    print(f"  Minimum grayscale mean: {gray_means.min():.2f}")
    print(f"  Maximum grayscale mean: {gray_means.max():.2f}")


def main():
    for label, (start, end) in IMAGE_TYPES.items():
        summarize_image_type(label, start, end)


if __name__ == "__main__":
    main()
