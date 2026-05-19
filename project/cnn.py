"""CNN training and inference for UNO symbol classification."""

import argparse
import json
import os
import random
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parent


DEFAULT_DATASET_DIR = PROJECT_DIR / "augm_symbols_dataset_1000_r75_25"
DEFAULT_MODEL_PATH = PROJECT_DIR / "symbol_cnn.pt"

IMAGE_SIZE = 64
MAX_ALLOWED_PARAMETERS = 12_000_000


def label_from_entry(entry):
    return entry["label"] if isinstance(entry, dict) else entry


def symbol_transform(image_size=IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


class SymbolDataset(Dataset):
    """Dataset used by the grader-side training script.

    It reads image paths from the json label files created for the symbol dataset
    and applies the same grayscale/resize/normalize preprocessing used in training.
    """

    def __init__(self, dataset_dir, class_to_idx, labels_file):
        dataset_dir = Path(dataset_dir)
        labels = json.loads((dataset_dir / labels_file).read_text())
        self.transform = symbol_transform()
        self.samples = []

        for filename, entry in sorted(labels.items()):
            image_path = dataset_dir / filename
            if image_path.exists():
                self.samples.append((image_path, class_to_idx[label_from_entry(entry)]))

        if not self.samples:
            raise ValueError(f"No labeled symbol images found in {dataset_dir / labels_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image), label


class SymbolCNN(nn.Module):
    """CNN for symbol classification."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class Classifier:
    """Inference wrapper used by `main.py`.

    It loads the saved checkpoint once, keeps the model in eval mode, and
    classifies batches of segmented symbol patches during csv generation.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="auto", cpu=False):
        self.device = choose_device(device, cpu)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SymbolCNN(len(checkpoint["class_to_idx"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = symbol_transform(checkpoint.get("image_size", IMAGE_SIZE))
        self.idx_to_class = checkpoint["idx_to_class"]
        self.checkpoint = checkpoint
        self.lock = threading.Lock()

    def classify_patches(self, patches):
        if not patches:
            return []

        tensors = []
        for patch in patches:
            image = Image.fromarray(np.asarray(patch)).convert("L")
            tensors.append(self.transform(image))

        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        with self.lock, torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)
        confidences, indices = probs.max(dim=1)

        results = []
        for index, confidence in zip(indices.cpu(), confidences.cpu()):
            results.append((self.idx_to_class[int(index)], float(confidence)))
        return results


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name="auto", cpu=False):
    if cpu:
        return torch.device("cpu")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch CUDA is not available.")
    return torch.device(device_name)


def load_class_mapping(dataset_dir, labels_files=("train.json", "test.json")):
    dataset_dir = Path(dataset_dir)
    labels = []

    for labels_file in labels_files:
        labels_path = dataset_dir / labels_file
        if labels_path.exists():
            entries = json.loads(labels_path.read_text())
            labels.extend(label_from_entry(entry) for entry in entries.values())

    if not labels:
        entries = json.loads((dataset_dir / "labels.json").read_text())
        labels.extend(label_from_entry(entry) for entry in entries.values())

    classes = sorted(set(labels), key=lambda value: (not value.isdigit(), value))
    return {label: idx for idx, label in enumerate(classes)}


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    use_amp = device.type == "cuda"
    scaler = getattr(run_epoch, "_scaler", None)

    if use_amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        run_epoch._scaler = scaler

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def default_num_workers(device):
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 0
    return min(8, cpu_count) if cpu_count > 0 else 0


def build_loader(dataset, batch_size, shuffle, num_workers, device):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    return DataLoader(**loader_kwargs)


def train_model(
    dataset_dir=DEFAULT_DATASET_DIR,
    train_labels="train.json",
    val_labels="test.json",
    output_path=DEFAULT_MODEL_PATH,
    epochs=10,
    batch_size=1024,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    num_workers=0,
    device="auto",
    cpu=False,
    show_confusion_matrix=True,
):
    """Train the symbol CNN and save the best validation checkpoint."""

    seed_everything(seed)
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    device = choose_device(device, cpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if num_workers <= 0:
        num_workers = default_num_workers(device)

    class_to_idx = load_class_mapping(dataset_dir, (train_labels, val_labels))
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    train_dataset = SymbolDataset(dataset_dir, class_to_idx, train_labels)
    val_dataset = SymbolDataset(dataset_dir, class_to_idx, val_labels)

    train_loader = build_loader(train_dataset, batch_size, True, num_workers, device)
    val_loader = build_loader(val_dataset, batch_size, False, num_workers, device)

    model = SymbolCNN(len(class_to_idx)).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count > MAX_ALLOWED_PARAMETERS:
        raise RuntimeError(f"Model has {parameter_count:,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"Dataset: {dataset_dir}")
    print(f"Training on {device} with {len(train_dataset)} train and {len(val_dataset)} val images")
    print(f"DataLoader workers: {num_workers}")
    print(f"Classes: {', '.join(class_to_idx.keys())}")
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

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
                    "image_size": IMAGE_SIZE,
                    "val_acc": val_acc,
                },
                output_path,
            )

    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Saved checkpoint to {output_path}")

    if show_confusion_matrix:
        print_confusion_matrix(output_path, val_loader, device)

    return output_path


def print_confusion_matrix(model_path, loader, device):
    model, checkpoint, _ = load_model(model_path, device=device)
    num_classes = len(checkpoint["class_to_idx"])
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1).cpu()
            for true_label, predicted_label in zip(labels, predictions):
                matrix[int(true_label), int(predicted_label)] += 1

    labels = [checkpoint["idx_to_class"][idx] for idx in range(num_classes)]
    width = max(5, len(str(int(matrix.max().item()))))
    name_width = max(len("true\\pred"), *(len(label) for label in labels))

    print("\nValidation confusion matrix")
    print("Rows=true labels, columns=predicted labels")
    print(" " * (name_width + 1) + " ".join(label.rjust(width) for label in labels))
    for row_index, label in enumerate(labels):
        row = " ".join(str(int(value)).rjust(width) for value in matrix[row_index])
        print(f"{label.rjust(name_width)} {row}")


def load_model(model_path=DEFAULT_MODEL_PATH, device=None):
    device = device or choose_device("auto")
    checkpoint = torch.load(model_path, map_location=device)
    model = SymbolCNN(len(checkpoint["class_to_idx"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def predict_image(image, model_path=DEFAULT_MODEL_PATH, device=None):
    classifier = Classifier(model_path, device=device or "auto")
    label, confidence = classifier.classify_patches([image])[0]
    return label, confidence


def parse_training_args():
    parser = argparse.ArgumentParser(description="Train the UNO symbol CNN.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train-labels", default="train.json")
    parser.add_argument("--val-labels", default="test.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-confusion-matrix", action="store_false", dest="show_confusion_matrix")
    parser.set_defaults(show_confusion_matrix=True)
    return parser.parse_args()


def run_training_cli():
    args = parse_training_args()
    train_model(
        dataset_dir=args.dataset_dir,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        cpu=args.cpu,
        show_confusion_matrix=args.show_confusion_matrix,
    )
