"""
notebook_utils.py
-----------------
Visualisation helpers used exclusively in the notebook.
Nothing here belongs in the training pipeline.
"""

import math
import matplotlib.pyplot as plt
from collections import Counter


def plot_distribution(name: str, labels: dict, ax: plt.Axes = None) -> plt.Axes:
    """
    Draw a bar chart of per-class image counts.

    Parameters
    ----------
    name   : chart title
    labels : {image_filename: class_label}
    ax     : existing Axes to draw on; a new figure is created if None

    Returns
    -------
    The Axes object.
    """
    counts  = Counter(labels.values())
    classes = sorted(counts.keys())
    values  = [counts[c] for c in classes]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 4))

    bars = ax.bar(classes, values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, max(values) * 1.18)
    return ax


def plot_distributions_grid(
    datasets: list[tuple[str, dict]],
    cols: int = 2,
) -> None:
    """
    Render bar charts for multiple label dicts side by side in a grid.

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

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def print_split_counts(
    train_labels: dict,
    val_labels: dict,
    train_target: int,
    val_target: int,
    num_classes: int,
    stage: str = "",
) -> None:
    """
    Print a summary table of image counts per split with expected totals.

    Parameters
    ----------
    train_labels  : {filename: label} for training
    val_labels    : {filename: label} for validation
    train_target  : expected images per class in train
    val_target    : expected images per class in validation
    num_classes   : number of symbol classes
    stage         : optional label prefix (e.g. "Before augmentation")
    """
    header = f"  {stage}" if stage else ""
    total_train = len(train_labels)
    total_val   = len(val_labels)
    exp_train   = train_target * num_classes
    exp_val     = val_target   * num_classes

    ok_train = "✓" if total_train == exp_train else "✗"
    ok_val   = "✓" if total_val   == exp_val   else "✗"

    print(f"\n{'─'*46}{header}")
    print(f"  {'Split':<12} {'Images':>8}  {'Expected':>8}  {'OK':>3}")
    print(f"  {'─'*12} {'─'*8}  {'─'*8}  {'─'*3}")
    print(f"  {'Train':<12} {total_train:>8}  {exp_train:>8}  {ok_train:>3}")
    print(f"  {'Validation':<12} {total_val:>8}  {exp_val:>8}  {ok_val:>3}")
    print(f"  {'TOTAL':<12} {total_train+total_val:>8}  {exp_train+exp_val:>8}")
    print(f"{'─'*46}")
