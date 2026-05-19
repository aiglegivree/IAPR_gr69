"""
augmentation_utils.py
---------------------
Pipeline utilities: I/O, dataset splitting, image copying,
label management, target computation, augmentation, and folder safety.

Import this in both the notebook and any training script.
"""

import os
import json
import random
import shutil
import math

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


def next_available_index(label_dicts: list[dict]) -> int:
    """
    Find the next free numeric index for new augmented filenames.

    Scans filenames of the form ``symbol_<N>.png`` across every dict in
    *label_dicts* and returns max(N) + 1.
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
# Target computation
# ---------------------------------------------------------------------------

def compute_targets(
    num_classes: int,
    total_train: int = 10000,
    val_ratio: float = 0.25,
) -> tuple[int, int]:
    """
    Compute per-class image targets for the train and validation splits.

    The train target is ``total_train // num_classes``.
    The validation target is derived so that the validation set is
    approximately ``val_ratio`` of the total dataset size.

    Parameters
    ----------
    num_classes : number of distinct symbol classes
    total_train : desired total number of training images
    val_ratio   : desired fraction of the full dataset for validation
                  (e.g. 0.25 → 25 %)

    Returns
    -------
    (train_target, val_target)  — per-class counts
    """
    train_target = total_train // num_classes
    # val_ratio = val / (train + val)  =>  val = train * ratio / (1 - ratio)
    val_per_class_exact = train_target * val_ratio / (1.0 - val_ratio)
    val_target = max(1, round(val_per_class_exact))
    return train_target, val_target


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def stratified_split(
    labels: dict,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Stratified train/validation split of an image-label mapping.

    Parameters
    ----------
    labels    : {image_filename: class_label}
    val_ratio : fraction of each class to put in the validation set
    seed      : random seed for reproducibility

    Returns
    -------
    (train_labels, val_labels) — same {filename: label} format
    """
    by_class = group_by_class(labels)
    train_labels: dict = {}
    val_labels:   dict = {}

    random.seed(seed)
    for label, images in by_class.items():
        random.shuffle(images)
        n_val = int(len(images) * val_ratio)
        for img in images[:n_val]:
            val_labels[img] = labels[img]
        for img in images[n_val:]:
            train_labels[img] = labels[img]

    return train_labels, val_labels


def copy_images(image_names: list[str], src_dir: str, dst_dir: str) -> None:
    """
    Copy a list of image files from *src_dir* to *dst_dir*.
    Already-present files are skipped silently.
    """
    to_copy = [
        n for n in image_names
        if not os.path.exists(os.path.join(dst_dir, n))
    ]
    for img_name in tqdm(to_copy, desc="Copying original images"):
        shutil.copy(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))


# ---------------------------------------------------------------------------
# Image augmentation
# ---------------------------------------------------------------------------

def augment_image(img: Image.Image) -> tuple[Image.Image, float, float, float]:
    """
    Apply a random rotation and small random translation to *img*.

    After the geometric transforms the result is re-binarized
    (thresholded at 128) to remove grey anti-aliasing fringing
    introduced by bilinear interpolation.

    Parameters
    ----------
    img : PIL Image (RGB or L)

    Returns
    -------
    (augmented_image, angle_deg, tx_pixels, ty_pixels)
    """
    angle = random.uniform(0, 360)   # symbols can appear in any orientation
    tx    = random.uniform(-10, 10)  # small shift to mimic off-centre capture
    ty    = random.uniform(-10, 10)

    aug = img.rotate(angle, resample=Image.BILINEAR)
    aug = aug.transform(
        img.size,
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.BILINEAR,
    )

    # Re-binarize: restores hard black/white look of originals
    aug_gray = aug.convert("L")
    aug_bin  = aug_gray.point(lambda p: 255 if p >= 128 else 0, "L")
    aug_out  = aug_bin.convert(aug.mode)

    return aug_out, angle, tx, ty


# ---------------------------------------------------------------------------
# Dataset completion (balancing via augmentation)
# ---------------------------------------------------------------------------

def complete_split(
    split_labels: dict,
    target_count: int,
    src_dir: str,
    out_dir: str,
    start_index: int,
    split_name: str = "split",
) -> tuple[dict, int]:
    """
    Balance one dataset split to *target_count* images per class.

    Only the *missing* images are generated — originals already present
    are counted and subtracted from the quota before any augmentation
    runs.  A single progress bar tracks the whole split.

    Parameters
    ----------
    split_labels : {image_filename: label}
    target_count : desired number of images per class after augmentation
    src_dir      : directory holding the original images (augmentation source)
    out_dir      : directory where new augmented images are saved
    start_index  : first free numeric index for new filenames
    split_name   : label shown in the tqdm progress bar

    Returns
    -------
    (updated_labels, next_free_index)
        updated_labels contains both originals and newly generated images.
    """
    new_labels = dict(split_labels)
    by_class   = group_by_class(split_labels)
    idx        = start_index

    total_needed = sum(
        max(0, target_count - len(imgs)) for imgs in by_class.values()
    )

    if total_needed == 0:
        print(f"[{split_name}] Already at target — no augmentation needed.")
        return new_labels, idx

    with tqdm(total=total_needed, desc=f"Augmenting {split_name}") as pbar:
        for cls, imgs in sorted(by_class.items()):
            needed = max(0, target_count - len(imgs))
            for _ in range(needed):
                src_name = random.choice(imgs)
                img = Image.open(os.path.join(src_dir, src_name)).convert("RGB")
                aug, _, _, _ = augment_image(img)
                new_name = f"symbol_{idx}.png"
                aug.save(os.path.join(out_dir, new_name))
                new_labels[new_name] = split_labels[src_name]
                idx += 1
                pbar.update(1)

    return new_labels, idx


# ---------------------------------------------------------------------------
# Output folder safety check
# ---------------------------------------------------------------------------

def prepare_output_dir(out_dir: str) -> bool:
    """
    Ensure the output directory exists and is ready to use.

    Returns
    -------
    True  — fresh run; caller should proceed with generation.
    False — folder already complete; caller should load from disk.

    Raises
    ------
    RuntimeError if the folder exists but is only partially populated
    (guards against a previous interrupted run).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        return True

    images, jsons = count_files(out_dir)
    has_splits = (
        os.path.exists(os.path.join(out_dir, "train.json")) and
        os.path.exists(os.path.join(out_dir, "val.json"))
    )

    if images > 0 and has_splits:
        print(
            f"[INFO] '{out_dir}' already contains {images} images and "
            "both split JSONs — skipping generation."
        )
        return False

    if images > 0 or jsons > 0:
        raise RuntimeError(
            f"'{out_dir}' exists but appears incomplete "
            f"({images} images, {jsons} JSONs). "
            "Delete it manually and re-run."
        )

    return True
