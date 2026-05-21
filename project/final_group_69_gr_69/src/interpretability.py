"""Interpretability helpers for the UNO symbol classifier notebook."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from cnn import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_PATH,
    SymbolDataset,
    choose_device,
    load_class_mapping,
    load_model,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DEFAULT_VAL_LABELS = "val.json"


def set_seed(seed: int = 42) -> None:
    """Make notebook sampling deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split_labels(dataset_dir=DEFAULT_DATASET_DIR, labels_file=DEFAULT_VAL_LABELS) -> dict:
    """Return a filename-to-label mapping for one split."""
    dataset_dir = Path(dataset_dir)
    labels_path = dataset_dir / labels_file
    return json.loads(labels_path.read_text())


def split_class_counts(labels: dict) -> dict:
    """Count how many samples belong to each class."""
    return dict(sorted(Counter(labels.values()).items(), key=lambda item: (not item[0].isdigit(), item[0])))


def build_symbol_datasets(dataset_dir=DEFAULT_DATASET_DIR, train_labels="train.json", val_labels=DEFAULT_VAL_LABELS):
    """Instantiate train and validation datasets with the saved class mapping."""
    dataset_dir = Path(dataset_dir)
    class_to_idx = load_class_mapping(dataset_dir, (train_labels, val_labels))
    train_dataset = SymbolDataset(dataset_dir, class_to_idx, train_labels)
    val_dataset = SymbolDataset(dataset_dir, class_to_idx, val_labels)
    return train_dataset, val_dataset, class_to_idx


def denormalize_symbol_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Undo grayscale normalization for display."""
    image = tensor.detach().cpu().clone()
    image = image * 0.5 + 0.5
    image = image.squeeze(0).clamp(0, 1).numpy()
    return image


def sample_examples_by_class(dataset, idx_to_class: dict, samples_per_class: int = 4, seed: int = 42) -> dict:
    """Pick a few dataset indices from each class."""
    rng = random.Random(seed)
    class_to_indices = defaultdict(list)
    for index, (_, label_idx) in enumerate(dataset.samples):
        class_to_indices[idx_to_class[label_idx]].append(index)

    sampled = {}
    for label in sorted(class_to_indices, key=lambda value: (not value.isdigit(), value)):
        indices = class_to_indices[label]
        count = min(samples_per_class, len(indices))
        sampled[label] = rng.sample(indices, count)
    return sampled


def plot_symbol_samples(dataset, idx_to_class: dict, samples_per_class: int = 4, seed: int = 42):
    """Display a compact class-balanced grid of symbol crops."""
    sampled = sample_examples_by_class(dataset, idx_to_class, samples_per_class=samples_per_class, seed=seed)
    labels = list(sampled)
    fig, axes = plt.subplots(len(labels), samples_per_class, figsize=(2.2 * samples_per_class, 1.9 * len(labels)))
    axes = np.atleast_2d(axes)

    for row, label in enumerate(labels):
        for col in range(samples_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(sampled[label]):
                image, _ = dataset[sampled[label][col]]
                ax.imshow(denormalize_symbol_tensor(image), cmap="gray")
            if col == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=18, fontsize=10)

    fig.suptitle("Representative symbol crops per class", y=1.0)
    fig.tight_layout()


def plot_first_layer_filters(model: torch.nn.Module, max_filters: int = 16):
    """Visualize the first convolution filters learned by the CNN."""
    weights = model.features[0].weight.detach().cpu().numpy()
    count = min(max_filters, weights.shape[0])
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for index, ax in enumerate(axes):
        ax.axis("off")
        if index < count:
            kernel = weights[index, 0]
            vmax = np.abs(kernel).max()
            ax.imshow(kernel, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"f{index}")

    fig.suptitle("First-layer convolution filters", y=1.02)
    fig.tight_layout()


def intermediate_layer_specs() -> list[tuple[str, int]]:
    """Return named checkpoints through the feature extractor."""
    return [
        ("after_first_relu", 2),
        ("after_block1_pool", 7),
        ("after_block2_pool", 15),
        ("after_block3_pool", 19),
    ]


@torch.no_grad()
def extract_intermediate_activations(model: torch.nn.Module, image_tensor: torch.Tensor, device="cpu") -> tuple[list[tuple[str, torch.Tensor]], torch.Tensor]:
    """Capture activations after a few key stages of the CNN."""
    model.eval()
    hooks = []
    captured = {}

    for name, layer_index in intermediate_layer_specs():
        def hook(_module, _inputs, output, layer_name=name):
            captured[layer_name] = output.detach().cpu()

        hooks.append(model.features[layer_index].register_forward_hook(hook))

    try:
        batch = image_tensor.unsqueeze(0).to(device)
        logits = model(batch).detach().cpu()[0]
    finally:
        for handle in hooks:
            handle.remove()

    activations = [(name, captured[name][0]) for name, _ in intermediate_layer_specs()]
    return activations, logits


def _normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    """Scale one activation map to [0, 1] for display."""
    feature_map = feature_map.astype(np.float32)
    feature_map -= feature_map.min()
    max_value = feature_map.max()
    if max_value > 0:
        feature_map /= max_value
    return feature_map


def top_activation_channels(activation: torch.Tensor, max_channels: int | None = 6) -> list[int]:
    """Pick channels by mean absolute response, or preserve native order when showing all."""
    scores = activation.abs().mean(dim=(1, 2))
    order = torch.argsort(scores, descending=True)
    if max_channels is None:
        return list(range(activation.shape[0]))
    return order[: min(max_channels, activation.shape[0])].tolist()


def plot_intermediate_activations(model, dataset, dataset_index: int, idx_to_class: dict, device="cpu", max_channels: int | None = None):
    """Visualize how one symbol is transformed across the CNN."""
    image_tensor, label = dataset[dataset_index]
    activations, logits = extract_intermediate_activations(model, image_tensor, device=device)
    probabilities = torch.softmax(logits, dim=0)
    predicted_index = int(probabilities.argmax())
    predicted_label = idx_to_class[predicted_index]
    confidence = float(probabilities[predicted_index])

    input_image = denormalize_symbol_tensor(image_tensor)
    for name, activation in activations:
        chosen_channels = top_activation_channels(activation, max_channels=max_channels)
        total_panels = 2 + len(chosen_channels)
        cols = min(8, total_panels)
        rows = int(np.ceil(total_panels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
        axes = np.atleast_1d(axes).ravel()

        axes[0].imshow(input_image, cmap="gray")
        axes[0].set_title(f"input\ntrue={idx_to_class[int(label)]}")
        axes[0].axis("off")

        mean_map = _normalize_feature_map(activation.mean(dim=0).numpy())
        axes[1].imshow(mean_map, cmap="magma")
        axes[1].set_title("mean map")
        axes[1].axis("off")

        for panel_index, channel_index in enumerate(chosen_channels, start=2):
            fmap = _normalize_feature_map(activation[channel_index].numpy())
            axes[panel_index].imshow(fmap, cmap="viridis")
            axes[panel_index].set_title(f"ch {channel_index}")
            axes[panel_index].axis("off")

        for ax in axes[total_panels:]:
            ax.axis("off")

        fig.suptitle(
            f"{name}\npred={predicted_label}  conf={confidence:.2f}  channels={len(chosen_channels)}",
            y=1.02,
        )
        fig.tight_layout()


def find_example_index_for_label(dataset, idx_to_class: dict, target_label: str) -> int:
    """Return the first dataset index matching a class label."""
    for dataset_index, (_, label_index) in enumerate(dataset.samples):
        if idx_to_class[label_index] == target_label:
            return dataset_index
    raise ValueError(f"Could not find label {target_label!r} in dataset.")


@torch.no_grad()
def collect_predictions(model, dataset, batch_size: int = 128, device="cpu"):
    """Run the classifier on a dataset and return logits, labels, and predictions."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_chunks = []
    labels_chunks = []

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        logits_chunks.append(logits.cpu())
        labels_chunks.append(labels.cpu())

    logits = torch.cat(logits_chunks)
    labels = torch.cat(labels_chunks)
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    return logits, labels, predictions, confidences


def confusion_matrix_from_predictions(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build a dense confusion matrix."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, predicted_label in zip(labels.tolist(), predictions.tolist()):
        matrix[true_label, predicted_label] += 1
    return matrix


@torch.no_grad()
def extract_embeddings(model, dataset, batch_size: int = 128, device="cpu"):
    """Return pooled CNN features together with labels and predictions."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    labels = []
    predictions = []

    model.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        features = model.features(images)
        pooled = model.classifier[0](features)
        flat = model.classifier[1](pooled)
        logits = model.classifier[3](flat)

        embeddings.append(flat.cpu())
        labels.append(batch_labels.cpu())
        predictions.append(logits.argmax(dim=1).cpu())

    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy(), torch.cat(predictions).numpy()


def project_embeddings(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project high-dimensional embeddings to 2D with PCA then t-SNE."""
    pca = PCA(n_components=min(30, embeddings.shape[1]), random_state=seed)
    reduced = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=20, random_state=seed)
    return tsne.fit_transform(reduced)


def plot_embedding_projection(points: np.ndarray, labels: np.ndarray, idx_to_class: dict):
    """Scatter-plot 2D embeddings colored by class."""
    class_names = [idx_to_class[index] for index in range(len(idx_to_class))]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", len(class_names))

    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        ax.scatter(points[mask, 0], points[mask, 1], s=22, alpha=0.75, color=cmap(class_index), label=class_name)

    ax.set_title("t-SNE projection of penultimate CNN features")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
    fig.tight_layout()


def grad_cam(model: torch.nn.Module, image_tensor: torch.Tensor, target_class: int, device="cpu") -> np.ndarray:
    """Compute a Grad-CAM heatmap on the last convolutional block."""
    model.eval()
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_forward = model.features.register_forward_hook(forward_hook)
    handle_backward = model.features.register_full_backward_hook(backward_hook)

    try:
        image_batch = image_tensor.unsqueeze(0).to(device)
        image_batch.requires_grad_(True)
        logits = model(image_batch)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()

        acts = activations["value"][0]
        grads = gradients["value"][0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        handle_forward.remove()
        handle_backward.remove()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resize and blend a heatmap over a grayscale symbol image."""
    image_uint8 = np.uint8(np.clip(image * 255, 0, 255))
    heatmap_uint8 = np.uint8(np.clip(heatmap * 255, 0, 255))
    heatmap_resized = np.array(Image.fromarray(heatmap_uint8).resize((image.shape[1], image.shape[0]), Image.Resampling.BILINEAR))
    colored = plt.cm.jet(heatmap_resized / 255.0)[..., :3]
    base = np.dstack([image_uint8 / 255.0] * 3)
    return np.clip((1 - alpha) * base + alpha * colored, 0, 1)


def top_confidence_examples(dataset, predictions: torch.Tensor, confidences: torch.Tensor, labels: torch.Tensor, per_class: int = 1) -> list[int]:
    """Return representative correctly classified indices."""
    chosen = []
    for class_index in sorted(labels.unique().tolist()):
        mask = (labels == class_index) & (predictions == class_index)
        class_indices = torch.where(mask)[0]
        if len(class_indices) == 0:
            continue
        sorted_indices = class_indices[torch.argsort(confidences[class_indices], descending=True)]
        chosen.extend(sorted_indices[:per_class].tolist())
    return chosen


def plot_gradcam_examples(model, dataset, idx_to_class: dict, labels: torch.Tensor, predictions: torch.Tensor, confidences: torch.Tensor, device="cpu", max_examples: int = 8):
    """Show Grad-CAM overlays for a few confidently correct samples."""
    selected_indices = top_confidence_examples(dataset, predictions, confidences, labels, per_class=1)[:max_examples]
    cols = 4
    rows = int(np.ceil(len(selected_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, dataset_index in zip(axes, selected_indices):
        image_tensor, label = dataset[dataset_index]
        heatmap = grad_cam(model, image_tensor, int(label), device=device)
        image = denormalize_symbol_tensor(image_tensor)
        overlay = overlay_heatmap(image, heatmap)
        ax.imshow(overlay)
        ax.set_title(f"true={idx_to_class[int(label)]}\nconf={float(confidences[dataset_index]):.2f}")
        ax.axis("off")

    for ax in axes[len(selected_indices):]:
        ax.axis("off")

    fig.suptitle("Grad-CAM on validation symbols", y=1.02)
    fig.tight_layout()


def load_scene_image(image_path) -> Image.Image:
    """Open a full UNO frame as RGB."""
    return Image.open(image_path).convert("RGB")


def classifier_summary(model: torch.nn.Module) -> dict:
    """Return a few compact architecture statistics."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "conv_blocks": 3,
    }


def load_classifier_components(model_path=DEFAULT_MODEL_PATH, device="auto"):
    """Load the saved classifier and metadata in notebook-friendly form."""
    actual_device = choose_device(device)
    model, checkpoint, actual_device = load_model(model_path=model_path, device=actual_device)
    return model, checkpoint, actual_device
