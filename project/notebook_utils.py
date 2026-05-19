import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


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
