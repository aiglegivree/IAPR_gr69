import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_DIR / "symbols_dataset"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "symbol_classification" / "symbol_cnn.pt"
IMAGE_SIZE = 64
MAX_ALLOWED_PARAMETERS = 12_000_000


class SymbolDataset(Dataset):
    def __init__(self, dataset_dir, class_to_idx):
        self.dataset_dir = Path(dataset_dir)
        self.class_to_idx = class_to_idx
        labels_path = self.dataset_dir / "labels.json"
        labels = json.loads(labels_path.read_text())

        self.samples = [
            (self.dataset_dir / filename, class_to_idx[label])
            for filename, label in sorted(labels.items())
            if (self.dataset_dir / filename).exists()
        ]

        if not self.samples:
            raise ValueError(f"No labeled symbol images found in {self.dataset_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = ImageOps.autocontrast(Image.open(image_path).convert("L"))
        return self.transform(image), label


class SymbolCNN(nn.Module):
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
        return self.classifier(self.features(x))


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_mapping(dataset_dir):
    labels = json.loads((Path(dataset_dir) / "labels.json").read_text())
    classes = sorted(set(labels.values()), key=lambda value: (not value.isdigit(), value))
    return {label: idx for idx, label in enumerate(classes)}


def stratified_split(dataset, val_fraction, seed):
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * val_fraction))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def compute_confusion_matrix(model, loader, num_classes, device):
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            for true_label, predicted_label in zip(labels, predictions):
                matrix[int(true_label), int(predicted_label)] += 1

    return matrix


def print_confusion_matrix(matrix, idx_to_class):
    labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]
    label_width = max(len(label) for label in labels + ["true\\pred"])
    cell_width = max(5, len(str(int(matrix.max().item()))))

    print("\nValidation confusion matrix")
    print("Rows=true labels, columns=predicted labels")
    header = " " * (label_width + 1) + " ".join(
        label.rjust(cell_width) for label in labels
    )
    print(header)

    for row_idx, label in enumerate(labels):
        values = " ".join(str(int(value)).rjust(cell_width) for value in matrix[row_idx])
        print(f"{label.rjust(label_width)} {values}")


def print_validation_scores(matrix, idx_to_class):
    true_positive = matrix.diag().float()
    support = matrix.sum(dim=1).float()
    predicted = matrix.sum(dim=0).float()

    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    accuracy = true_positive.sum() / matrix.sum().clamp_min(1).float()
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / support.sum().clamp_min(1)

    print("\nValidation scores")
    print(f"accuracy:    {accuracy.item():.3f}")
    print(f"macro_f1:    {macro_f1.item():.3f}")
    print(f"weighted_f1: {weighted_f1.item():.3f}")

    print("\nPer-class validation scores")
    print("   label precision recall     f1 support")
    for idx in range(len(idx_to_class)):
        print(
            f"{idx_to_class[idx]:>8} "
            f"{precision[idx].item():9.3f} "
            f"{recall[idx].item():6.3f} "
            f"{f1[idx].item():6.3f} "
            f"{int(support[idx].item()):7d}"
        )


def choose_device(args):
    if args.cpu:
        return torch.device("cpu")

    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch install cannot use it. "
            "Your GPU is visible to nvidia-smi, so install a CUDA-enabled PyTorch build."
        )

    return torch.device(args.device)


def train(args):
    seed_everything(args.seed)
    device = choose_device(args)

    class_to_idx = load_class_mapping(args.dataset_dir)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    dataset = SymbolDataset(args.dataset_dir, class_to_idx)
    train_indices, val_indices = stratified_split(dataset, args.val_fraction, args.seed)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SymbolCNN(num_classes=len(class_to_idx)).to(device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count > MAX_ALLOWED_PARAMETERS:
        raise RuntimeError(
            f"Model has {parameter_count:,} trainable parameters, "
            f"which exceeds the {MAX_ALLOWED_PARAMETERS:,} parameter limit."
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device} with {len(train_indices)} train and {len(val_indices)} val images")
    print(f"Classes: {', '.join(class_to_idx.keys())}")
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        print(
            f"epoch {epoch:02d}/{args.epochs} "
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
                args.output,
            )

    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Saved checkpoint to {args.output}")

    if args.show_confusion_matrix:
        checkpoint = torch.load(args.output, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        matrix = compute_confusion_matrix(model, val_loader, len(class_to_idx), device)
        print_confusion_matrix(matrix, idx_to_class)
        print_validation_scores(matrix, idx_to_class)


def load_model(checkpoint_path, device=None):
    device = device or torch.device("cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch install cannot use it. "
            "Install a CUDA-enabled PyTorch build or pass device=torch.device('cpu')."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SymbolCNN(num_classes=len(checkpoint["class_to_idx"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def predict_image(image, checkpoint_path=DEFAULT_OUTPUT_PATH):
    model, checkpoint, device = load_model(checkpoint_path)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((checkpoint.get("image_size", IMAGE_SIZE),) * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    tensor = transform(ImageOps.autocontrast(image.convert("L"))).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, class_idx = probabilities.max(dim=0)
    return checkpoint["idx_to_class"][int(class_idx)], float(confidence)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN for UNO symbol classification.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--cpu", action="store_true", help="Shortcut for --device cpu.")
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_false",
        dest="show_confusion_matrix",
        help="Do not print the validation confusion matrix after training.",
    )
    parser.set_defaults(show_confusion_matrix=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
