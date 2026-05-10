from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_COLORS = ("blue", "red", "yellow", "black", "green")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CardDetectionPaths:
    base_dir: Path
    dataset_name: str = "iapr-26-uno-vision-challenge"

    @property
    def dataset_dir(self) -> Path:
        return self.base_dir / self.dataset_name

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "train_images"

    @property
    def reference_dir(self) -> Path:
        return self.dataset_dir / "reference_images"

    @property
    def reference(self) -> Path:
        return self.reference_dir

    @property
    def cards_dir(self) -> Path:
        return self.base_dir / "cards"

    @property
    def threshold_rectangle_settings(self) -> Path:
        return self.base_dir / "threshold_rectangle_set.txt"

    @property
    def rectangle_settings(self) -> Path:
        return self.base_dir / "hough_set.txt"

    @property
    def card_template_mask(self) -> Path:
        return self.base_dir / "general_card_mask.png"

    def hsv_file(self, color: str) -> Path:
        return self.base_dir / f"{color}_hsv.txt"

    def gray_file(self, color: str) -> Path:
        return self.base_dir / f"{color}_gray.txt"
