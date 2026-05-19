"""
dataset_utils.py
----------------
Dataset management utilities for the augmentation pipeline:
  - I/O (JSON, file counting)
  - train/test splitting
  - image copying
  - label grouping and target computation
  - dataset completion (augmentation loop)
  - bar-plot visualisations
"""

import os
import json
import math
import random
import shutil

import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from PIL import Image
from tqdm import tqdm

from augmentation_utils import augment_image


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
# Target computation
# ---------------------------------------------------------------------------

def compute_targets(
    num_classes: int,
    total_train: int = 6000,
    test_ratio: float = 0.1,
) -> tuple[int, int]:
    """
    Compute per-class image targets for the train and test splits.

    The train target is simply ``total_train // num_classes``.
    The test target is derived so that the test set is approximately
    ``test_ratio`` of the total dataset size — rounded to the nearest
    integer.

    Parameters
    ----------
    num_classes : number of distinct symbol classes
    total_train : desired total number of training images
    test_ratio  : desired fraction of the full dataset reserved for test
                  (e.g. 0.1 → ~10 %)

    Returns
    -------
    (train_target, test_target)  — per-class counts
    """
    train_target = total_train // num_classes
    # test_ratio = test / (train + test)  =>  test = train * ratio / (1 - ratio)
    test_per_class_exact = train_target * test_ratio / (1.0 - test_ratio)
    test_target = max(1, round(test_per_class_exact))
    return train_target, test_target


# ---------------------------------------------------------------------------
# Dataset preparation
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
    (train_labels, test_labels) — same {filename: label} format
    """
    by_class = group_by_class(labels)
    train_labels: dict = {}
    test_labels:  dict = {}

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
    for img_name in tqdm(image_names, desc="Copying original images"):
        src = os.path.join(src_dir, img_name)
        dst = os.path.join(dst_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)


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
    Balance one dataset split (train *or* test) to *target_count* per class.

    For every class that has fewer than *target_count* images, augmented
    copies are generated from randomly chosen originals and saved to
    *out_dir*.  A single tqdm bar tracks total progress for the whole
    split (not one bar per class).

    Parameters
    ----------
    split_labels : {image_filename: label}
    target_count : desired number of images per class
    src_dir      : directory holding the *original* images
    out_dir      : directory where augmented images are saved
    start_index  : first free numeric index for new filenames
    split_name   : label shown in the tqdm progress bar

    Returns
    -------
    (updated_labels, next_free_index)
    """
    new_labels = dict(split_labels)
    by_class   = group_by_class(split_labels)
    idx        = start_index

    # Pre-compute total images to generate across all classes
    total_needed = sum(
        max(0, target_count - len(imgs)) for imgs in by_class.values()
    )

    with tqdm(total=total_needed, desc=f"Augmenting {split_name}") as pbar:
        for cls, imgs in by_class.items():
            current = len(imgs)
            needed  = max(0, target_count - current)

            for _ in range(needed):
                src_name = random.choice(imgs)
                img = Image.open(os.path.join(src_dir, src_name)).convert("RGB")

                aug, _angle, _tx, _ty = augment_image(img)

                new_name = f"symbol_{idx}.png"
                aug.save(os.path.join(out_dir, new_name))
                new_labels[new_name] = split_labels[src_name]
                idx  += 1
                pbar.update(1)

    return new_labels, idx


# ---------------------------------------------------------------------------
# Output folder safety check
# ---------------------------------------------------------------------------

def prepare_output_dir(out_dir: str) -> bool:
    """
    Ensure the output directory exists and is ready to use.

    If the directory already exists **and** already contains images and
    JSON splits, it is considered complete and the function returns
    ``False`` (caller should skip re-generation).  If it does not exist,
    it is created and ``True`` is returned.

    Raises
    ------
    RuntimeError
        If the directory exists but is only partially populated (safety
        guard against a previous interrupted run).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        return True  # fresh run

    images, jsons = count_files(out_dir)
    has_splits = (
        os.path.exists(os.path.join(out_dir, "train.json")) and
        os.path.exists(os.path.join(out_dir, "test.json"))
    )

    if images > 0 and has_splits:
        print(
            f"[INFO] '{out_dir}' already contains {images} images and "
            f"both split JSONs — skipping generation."
        )
        return False  # already done

    if images > 0 or jsons > 0:
        raise RuntimeError(
            f"'{out_dir}' exists but appears incomplete "
            f"({images} images, {jsons} JSONs). "
            "Delete it manually and re-run."
        )

    return True  # directory exists but is empty


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distribution(name: str, labels: dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Draw a bar chart of per-class image counts.

    Parameters
    ----------
    name   : title for the chart
    labels : {image_filename: class_label}
    ax     : existing Axes to draw on; a new figure is created if None

    Returns
    -------
    The Axes object (so the caller can further customise or save it).
    """
    counts = Counter(labels.values())
    classes = sorted(counts.keys())
    values  = [counts[c] for c in classes]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7), 4))

    bars = ax.bar(classes, values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, max(values) * 1.15)
    return ax


def plot_distributions_grid(
    datasets: list[tuple[str, dict]],
    cols: int = 2,
) -> None:
    """
    Plot bar charts for multiple label dicts in a grid.

    Parameters
    ----------
    datasets : list of (title, labels_dict)
    cols     : number of columns in the grid
    """
    rows = math.ceil(len(datasets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (name, labels) in zip(axes, datasets):
        plot_distribution(name, labels, ax=ax)

    # Hide unused subplots
    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
