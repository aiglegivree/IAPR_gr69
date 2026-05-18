import json

import cv2
from PIL import Image

import segmentation


INPUT_IMAGE_DIR = segmentation.TRAIN_DIR
SYMBOL_DATASET_DIR = segmentation.BASE_DIR / "symbols_dataset"
LABELS_PATH = SYMBOL_DATASET_DIR / "labels.json"


def reset_symbol_dataset_dir():
    SYMBOL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for old_symbol_path in SYMBOL_DATASET_DIR.glob("symbol_*.png"):
        old_symbol_path.unlink()

    with LABELS_PATH.open("w") as file:
        json.dump({}, file, indent=2, sort_keys=True)


def show_progress(current, total, width=30):
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}", end="", flush=True)

    if current == total:
        print()


def build_symbol_dataset():
    reset_symbol_dataset_dir()

    acceleration_backend = segmentation.configure_acceleration()
    segmentation.set_acceleration_backend(acceleration_backend)
    print(f"Using {acceleration_backend} acceleration for dataset build")

    symbol_number = 1
    image_paths = sorted(INPUT_IMAGE_DIR.glob("*.jpg"))
    total_images = len(image_paths)

    for image_idx, image_path in enumerate(image_paths, start=1):
        show_progress(image_idx, total_images)
        img_color = Image.open(image_path).convert("RGB")
        detected_symbols = segmentation.detect_symbols(img_color)

        for symbol_patch, _, _ in detected_symbols:
            output_path = SYMBOL_DATASET_DIR / f"symbol_{symbol_number}.png"
            cv2.imwrite(str(output_path), symbol_patch)
            symbol_number += 1

    print(f"Saved {symbol_number - 1} symbols to {SYMBOL_DATASET_DIR}")


if __name__ == "__main__":
    build_symbol_dataset()
