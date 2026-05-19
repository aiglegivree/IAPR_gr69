"""
augmentation_utils.py
---------------------
All reusable functions for the symbol-dataset augmentation pipeline.
"""

import os
import json
import random
import shutil

from collections import Counter, defaultdict
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    """Load a JSON file and return its contents as a dict."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save a dict as a formatted JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def count_files(folder: str) -> tuple[int, int]:
    """
    Count image files and JSON files in *folder* (non-recursive).

    Returns
    -------
    (num_images, num_json)
    """
    num_images = num_json = 0
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            if file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                num_images += 1
            elif file.endswith(".json"):
                num_json += 1
    return num_images, num_json


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def stratified_split(
    labels: dict,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Stratified train/test split of an image-label mapping.

    Parameters
    ----------
    labels     : {image_filename: class_label}
    test_ratio : fraction of each class to put in the test set
    seed       : random seed for reproducibility

    Returns
    -------
    (train_labels, test_labels) — same format as *labels*
    """
    by_class = group_by_class(labels)
    train_labels: dict = {}
    test_labels: dict = {}

    random.seed(seed)
    for label, images in by_class.items():
        random.shuffle(images)
        n_test = int(len(images) * test_ratio)
        for img in images[:n_test]:
            test_labels[img] = labels[img]
        for img in images[n_test:]:
            train_labels[img] = labels[img]

    return train_labels, test_labels


def copy_images(image_names: list[str], src_dir: str, dst_dir: str) -> None:
    """
    Copy a list of image files from *src_dir* to *dst_dir*.
    Already-present files are skipped silently.
    """
    for img_name in tqdm(image_names, desc="Copying images"):
        src = os.path.join(src_dir, img_name)
        dst = os.path.join(dst_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)


# ---------------------------------------------------------------------------
# Label / class utilities
# ---------------------------------------------------------------------------

def group_by_class(labels: dict) -> defaultdict:
    """
    Invert a {image: label} mapping into {label: [image, ...]}.

    Returns a defaultdict(list).
    """
    d: defaultdict = defaultdict(list)
    for img, label in labels.items():
        d[label].append(img)
    return d


def print_distribution(name: str, labels: dict) -> None:
    """Print per-class counts and total for a label dict."""
    c = Counter(labels.values())
    total = sum(c.values())
    print(f"\n{name}")
    for k in sorted(c.keys()):
        print(f"  {k}: {c[k]}")
    print(f"  TOTAL: {total}")


def next_available_index(label_dicts: list[dict]) -> int:
    """
    Find the next free numeric index for new augmented filenames.

    Scans all filenames of the form ``symbol_<N>.png`` across every
    dict in *label_dicts* and returns max(N) + 1.
    """
    max_index = 0
    for labels in label_dicts:
        for f in labels:
            try:
                idx = int(f.replace("symbol_", "").replace(".png", ""))
                max_index = max(max_index, idx)
            except ValueError:
                pass
    return max_index + 1


# ---------------------------------------------------------------------------
# Image augmentation
# ---------------------------------------------------------------------------

def augment_image(img: Image.Image) -> tuple[Image.Image, float, float, float]:
    """
    Apply a random rotation and small random translation to *img*.

    After geometric transforms the result is re-binarized (thresholded
    at 128) to undo the grey fringing introduced by bilinear
    interpolation — keeping the same hard black/white look as the
    originals.

    Parameters
    ----------
    img : PIL Image (RGB or L)

    Returns
    -------
    (augmented_image, angle_deg, tx_pixels, ty_pixels)
    """
    angle = random.uniform(0, 360)   # full rotation: digits can appear in any orientation
    tx = random.uniform(-10, 10)     # small shift to mimic off-centre capture
    ty = random.uniform(-10, 10)

    aug = img.rotate(angle, resample=Image.BILINEAR)
    aug = aug.transform(
        img.size,
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.BILINEAR,
    )

    # Re-binarize: bilinear interpolation creates grey anti-aliasing
    # artefacts; thresholding at 128 restores a clean binary image.
    aug_gray = aug.convert("L")
    aug_bin = aug_gray.point(lambda p: 255 if p >= 128 else 0, "L")
    aug_out = aug_bin.convert(aug.mode)   # restore original colour mode

    return aug_out, angle, tx, ty


# ---------------------------------------------------------------------------
# Dataset completion (balancing)
# ---------------------------------------------------------------------------

def complete_split(
    split_labels: dict,
    target_count: int,
    src_dir: str,
    out_dir: str,
    start_index: int,
) -> tuple[dict, int]:
    """
    Balance one dataset split (train *or* test) to *target_count* per class.

    For every class that has fewer than *target_count* images, new
    augmented images are generated from randomly chosen originals and
    saved to *out_dir*.

    Parameters
    ----------
    split_labels  : {image_filename: label} — the split to complete
    target_count  : desired number of images per class
    src_dir       : directory that holds the original images
    out_dir       : directory where augmented images are saved
    start_index   : first free numeric index for new filenames

    Returns
    -------
    (updated_labels, next_free_index)
        updated_labels includes both originals and newly generated images.
    """
    new_labels = dict(split_labels)
    by_class = group_by_class(split_labels)
    idx = start_index

    for cls, imgs in by_class.items():
        current = len(imgs)
        needed = max(0, target_count - current)
        print(f"  Class {cls}: {current} → adding {needed}")

        for _ in tqdm(range(needed), leave=False):
            src_name = random.choice(imgs)
            img = Image.open(os.path.join(src_dir, src_name)).convert("RGB")

            aug, _angle, _tx, _ty = augment_image(img)

            new_name = f"symbol_{idx}.png"
            aug.save(os.path.join(out_dir, new_name))
            new_labels[new_name] = split_labels[src_name]
            idx += 1

    return new_labels, idx


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_stats(name: str, labels: dict, image_dir: str) -> None:
    """
    Print a summary of *labels* (from JSON) and the actual image count
    in *image_dir*.
    """
    label_counts = Counter(labels.values())
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    print(f"\n{'='*40}")
    print(f"{name} — FINAL STATS")
    print(f"{'='*40}")
    print("Labels (from JSON):")
    for k in sorted(label_counts.keys()):
        print(f"  {k}: {label_counts[k]}")
    print(f"  TOTAL LABELS : {sum(label_counts.values())}")
    print(f"  TOTAL IMAGES : {len(image_files)}")
