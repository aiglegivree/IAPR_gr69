import importlib.util
import json
from collections import defaultdict
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from cnn import *
from clustering import *


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "iapr-26-uno-vision-challenge"
SEGMENTATION_DIR = PROJECT_DIR / "segmentation"
DEFAULT_TEST_IMAGES_DIR = DATA_DIR / "test_images"
DEFAULT_HISTORY_PATH = PROJECT_DIR / "symbol_cnn_history.json"

SEGMENTATION_FILE = SEGMENTATION_DIR / "segmentation_utils.py"
SEGMENTATION_SPEC = importlib.util.spec_from_file_location("uno_segmentation_module", SEGMENTATION_FILE)
segmentation = importlib.util.module_from_spec(SEGMENTATION_SPEC)
SEGMENTATION_SPEC.loader.exec_module(segmentation)

def show_image(image, title=None, ax=None):
    """Display one RGB image and hide notebook plot axes."""
    ax = ax or plt.subplots(figsize=(10, 8))[1]
    ax.imshow(image)
    ax.axis("off")

    if title:
        ax.set_title(title)

    return ax


def show_masks(masks, columns=3, title=None):
    """Display a dictionary of binary masks in a compact grid."""
    names = list(masks)
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        ax.imshow(masks[name], cmap="gray")
        ax.set_title(name.capitalize())
        ax.axis("off")

    for ax in axes[len(names):]:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()


def overlay_mask(image, mask, color=(255, 0, 255), title=None):
    """Display an image with foreground mask pixels highlighted in one color."""
    overlay = np.array(image).copy()
    overlay[mask > 0] = color
    return show_image(overlay, title)




def ensure_model(model_path=DEFAULT_MODEL_PATH, history_path=DEFAULT_HISTORY_PATH, **train_kwargs):
    """Train the CNN only if the model file does not already exist."""

    model_path = Path(model_path)
    history_path = Path(history_path)
    trained_now = False

    if not model_path.exists():
        train_with_history(output_path=model_path, history_path=history_path, **train_kwargs)
        trained_now = True

    history = load_history(history_path)
    return model_path, history_path, history, trained_now


def load_history(history_path=DEFAULT_HISTORY_PATH):
    """Load the saved training history if it exists."""

    history_path = Path(history_path)
    if not history_path.exists():
        return None
    return json.loads(history_path.read_text())


def train_with_history(
    dataset_dir=DEFAULT_DATASET_DIR,
    train_labels="train.json",
    val_labels="val.json",
    output_path=DEFAULT_MODEL_PATH,
    history_path=DEFAULT_HISTORY_PATH,
    epochs=10,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    num_workers=0,
    device="auto",
    cpu=False,
):
    """Train the same CNN as cnn.py and save a side history file for plots."""

    seed_everything(seed)
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    history_path = Path(history_path)
    device = choose_device(device, cpu)
    if not (dataset_dir / val_labels).exists() and val_labels == "val.json":
        val_labels = "test.json"

    class_to_idx = load_class_mapping(dataset_dir, (train_labels, val_labels))
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    train_dataset = SymbolDataset(dataset_dir, class_to_idx, train_labels)
    val_dataset = SymbolDataset(dataset_dir, class_to_idx, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SymbolCNN(len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "image_size": 64,
                    "val_acc": val_acc,
                },
                output_path,
            )

    history_path.write_text(json.dumps(history, indent=2))
    return output_path, history


def evaluate_model(
    model_path=DEFAULT_MODEL_PATH,
    dataset_dir=DEFAULT_DATASET_DIR,
    labels_file="val.json",
    batch_size=64,
    num_workers=0,
    device="auto",
    cpu=False,
):
    """Compute accuracy, F1 score, and confusion matrix on one split."""

    device = choose_device(device, cpu)
    model, checkpoint, device = load_model(model_path, device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]
    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / labels_file).exists() and labels_file == "val.json":
        labels_file = "test.json"
    dataset = SymbolDataset(dataset_dir, class_to_idx, labels_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    matrix = torch.zeros((len(class_to_idx), len(class_to_idx)), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1).cpu()
            for true_label, predicted_label in zip(labels, predictions):
                matrix[int(true_label), int(predicted_label)] += 1

    tp = matrix.diag().float()
    support = matrix.sum(dim=1).float()
    predicted = matrix.sum(dim=0).float()
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    metrics = {
        "accuracy": (tp.sum() / matrix.sum().clamp_min(1).float()).item(),
        "f1": f1.mean().item(),
    }
    return metrics, matrix, idx_to_class


def plot_history(history):
    """Plot training/validation accuracy and loss."""

    if history is None:
        print("No saved training history was found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_acc"], label="train")
    axes[0].plot(epochs, history["val_acc"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()


def plot_confusion_matrix(matrix, idx_to_class):
    """Display the confusion matrix as a heatmap."""

    labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()


def build_pipeline_debug(image_path, classifier, eps=525, buffer=200):
    """Collect all intermediate results for one full pipeline run."""

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image)
    height, width = image_rgb.shape[:2]

    records = detect_and_classify(image_rgb, classifier)
    assign_players(records, width, height, eps=eps, buffer=buffer)
    token_center = segmentation.get_token_center(image)
    row = prediction_row_for_image(image_path, classifier, eps=eps, buffer=buffer)

    grouped = defaultdict(list)
    collapsed = {}
    for record in records:
        player_id = record.get("player_id")
        if player_id is not None:
            grouped[player_id].append(record)

    kept_by_player = {}
    for player_id in PLAYER_ID_TO_COLUMN:
        player_records = sort_player_cards(grouped.get(player_id, []), player_id)
        after_cards = collapse_duplicate_detections(player_records)
        kept_records = []
        remaining = list(after_cards)
        for record in player_records:
            if record["card"] in remaining:
                kept_records.append(record)
                remaining.remove(record["card"])
        kept_by_player[player_id] = kept_records
        collapsed[player_id] = {"before": [record["card"] for record in player_records], "after": after_cards}

    return {
        "image_path": image_path,
        "image_rgb": image_rgb,
        "records": records,
        "token_center": token_center,
        "width": width,
        "height": height,
        "row": row,
        "grouped": grouped,
        "kept_by_player": kept_by_player,
        "collapsed": collapsed,
        "active_player": active_player_from_token(token_center, width, height, buffer=buffer),
    }


def show_patch_predictions(image_path, classifier, max_patches=12):
    """Show the first detected symbol patches and their CNN predictions."""

    debug = build_pipeline_debug(image_path, classifier)
    records = debug["records"][:max_patches]
    if not records:
        print("No symbol patches were detected on this image.")
        return debug

    cols = min(4, len(records))
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, record in zip(axes, records):
        ax.imshow(record["patch"], cmap="gray")
        ax.set_title(f"{record['card']}\n{record['confidence']:.2f}")
        ax.axis("off")

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle(f"Detected symbol patches for {Path(image_path).name}")
    fig.tight_layout()
    return debug


def show_token_assignment(debug, buffer=200):
    """Plot token position and player anchor positions."""

    image_rgb = debug["image_rgb"]
    width = debug["width"]
    height = debug["height"]
    anchors = np.array(
        [
            [width // 2, height - buffer],
            [width - buffer, height // 2],
            [width // 2, buffer],
            [buffer, height // 2],
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image_rgb)
    names = ["p1", "p2", "p3", "p4"]
    for name, (x, y) in zip(names, anchors):
        ax.scatter(x, y, s=140, marker="x")
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + 96, y - 70),
            color="black",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "#ffe680", "edgecolor": "black", "alpha": 0.95, "pad": 3},
            arrowprops={"arrowstyle": "-", "color": "#d9b300", "lw": 1.8},
        )

    if debug["token_center"] is not None:
        x, y = debug["token_center"]
        ax.scatter(x, y, s=180, marker="o", edgecolors="yellow", facecolors="none", linewidths=2)
        ax.annotate(
            f"token -> {debug['active_player']}",
            xy=(x, y),
            xytext=(x + 70, y - 60),
            color="yellow",
            fontsize=12,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            arrowprops={"arrowstyle": "-", "color": "yellow", "lw": 1.4},
        )

    ax.set_title("Token attribution to the active player")
    ax.axis("off")


def show_pipeline_assignments(debug, show_image=True, remove_duplicates=False, show_anchors=False, buffer=200):
    """Plot center/player assignments and detected card labels.

    If show_image is False, the assignments are drawn on a blank canvas instead
    of on top of the original frame.
    """

    fig, ax = plt.subplots(figsize=(10, 7))
    if show_image:
        ax.imshow(debug["image_rgb"])
    else:
        blank = np.ones_like(debug["image_rgb"]) * 255
        ax.imshow(blank)
    colors = {0: "cyan", 1: "lime", 2: "orange", 3: "magenta", 4: "red", None: "white"}

    records_to_plot = []
    if remove_duplicates:
        center_records = [record for record in debug["records"] if record.get("player_id") == 0]
        if center_records:
            best_center = max(center_records, key=lambda record: record["confidence"])
            records_to_plot.append(best_center)
        for kept_records in debug["kept_by_player"].values():
            records_to_plot.extend(kept_records)
    else:
        records_to_plot = debug["records"]

    for index, record in enumerate(records_to_plot):
        x, y = record["center"]
        player_id = record.get("player_id")
        label = record["card"]
        color = colors.get(player_id, "white")
        ax.scatter(x, y, c=color, s=70)
        ax.annotate(
            f"{label} | p{player_id}" if player_id is not None else label,
            xy=(x, y),
            xytext=(x + 52, y - 44),
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
        )

    width = debug["width"]
    height = debug["height"]
    anchors = {
        "center": (width / 2, height / 2),
        "p1": (width / 2, height - buffer),
        "p2": (width - buffer, height / 2),
        "p3": (width / 2, buffer),
        "p4": (buffer, height / 2),
    }
    for name, (x, y) in anchors.items():
        marker_color = "cyan" if name == "center" else "white"
        ax.scatter(x, y, s=150, marker="x", c=marker_color, linewidths=2)
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + 96, y - 70),
            color="black",
            fontsize=11,
            fontweight="bold",
            bbox={
                "facecolor": "#8fe8ff" if name == "center" else "#ffe680",
                "edgecolor": "black",
                "alpha": 0.95,
                "pad": 3,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#00bcd4" if name == "center" else "#d9b300",
                "lw": 1.8,
            },
        )

    ax.set_title("Detected cards with center/player attribution")
    ax.axis("off")


def show_full_pipeline_frame(debug, buffer=200, remove_duplicates=False):
    """Show one combined frame with card attribution and token attribution."""

    image_rgb = debug["image_rgb"]
    width = debug["width"]
    height = debug["height"]
    anchors = np.array(
        [
            [width // 2, height - buffer],
            [width - buffer, height // 2],
            [width // 2, buffer],
            [buffer, height // 2],
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image_rgb)

    colors = {0: "cyan", 1: "lime", 2: "orange", 3: "magenta", 4: "red", None: "white"}
    records_to_plot = []
    if remove_duplicates:
        center_records = [record for record in debug["records"] if record.get("player_id") == 0]
        if center_records:
            records_to_plot.append(max(center_records, key=lambda record: record["confidence"]))
        for kept_records in debug["kept_by_player"].values():
            records_to_plot.extend(kept_records)
    else:
        records_to_plot = debug["records"]

    for record in records_to_plot:
        x, y = record["center"]
        player_id = record.get("player_id")
        label = record["card"]
        color = colors.get(player_id, "white")
        ax.scatter(x, y, c=color, s=70)
        ax.annotate(
            f"{label} | p{player_id}" if player_id is not None else label,
            xy=(x, y),
            xytext=(x + 52, y - 44),
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
        )

    anchor_points = {
        "center": (width / 2, height / 2),
        "p1": (width / 2, height - buffer),
        "p2": (width - buffer, height / 2),
        "p3": (width / 2, buffer),
        "p4": (buffer, height / 2),
    }
    for name, (x, y) in anchor_points.items():
        marker_color = "cyan" if name == "center" else "white"
        box_color = "#8fe8ff" if name == "center" else "#ffe680"
        line_color = "#00bcd4" if name == "center" else "#d9b300"
        ax.scatter(x, y, s=150, marker="x", c=marker_color, linewidths=2)
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x + 96, y - 70),
            color="black",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": box_color, "edgecolor": "black", "alpha": 0.95, "pad": 3},
            arrowprops={"arrowstyle": "-", "color": line_color, "lw": 1.8},
        )

    if debug["token_center"] is not None:
        x, y = debug["token_center"]
        ax.scatter(x, y, s=180, marker="o", edgecolors="yellow", facecolors="none", linewidths=2)
        ax.annotate(
            f"token -> {debug['active_player']}",
            xy=(x, y),
            xytext=(x + 70, y - 60),
            color="yellow",
            fontsize=12,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            arrowprops={"arrowstyle": "-", "color": "yellow", "lw": 1.4},
        )

    ax.set_title("Full pipeline result on one frame")
    ax.axis("off")


def show_duplicate_collapse(debug):
    """Print the before/after card lists used for duplicate deletion."""

    for player_id, info in debug["collapsed"].items():
        print(f"Player {player_id}")
        print("  before:", info["before"] if info["before"] else ["EMPTY"])
        print("  after: ", info["after"] if info["after"] else ["EMPTY"])


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
