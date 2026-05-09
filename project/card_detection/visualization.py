from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import CardDetectionResult


def show_images(images: list[tuple[str, np.ndarray]], cols: int = 3, figsize: tuple[int, int] = (14, 10), empty_message: str = "No images to show") -> None:
    if not images:
        print(empty_message)
        return
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_detection_result(result: CardDetectionResult) -> None:
    show_images(result.debug_views, cols=3, figsize=(14, 16), empty_message="No debug masks to show")
    show_images([("detections", result.overlay)], cols=1, figsize=(8, 8))
    show_images(result.cards, cols=4, figsize=(14, 10), empty_message="No cards detected")
