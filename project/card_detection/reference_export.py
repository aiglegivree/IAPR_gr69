from __future__ import annotations

import re
from pathlib import Path

import cv2
import ipywidgets as widgets
from IPython.display import display

from .config import DEFAULT_COLORS, CardDetectionPaths
from .io import load_image
from .pipeline import detect_cards_in_image


LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_card_label(label: str, fallback: str) -> str:
    cleaned = LABEL_PATTERN.sub("_", label.strip()).strip("._-")
    return cleaned or fallback


def create_reference_card_labeler(
    paths: CardDetectionPaths,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask,
    scale: float = 0.25,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    output_dir: Path | None = None,
):
    output_dir = output_dir or paths.cards_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = extract_reference_card_groups(paths, color_thresholds, rectangle_settings, card_template_mask, scale, colors)
    items = [item for group in groups for item in group["items"]]
    save_output = widgets.Output()

    def save_cards(_button=None):
        saved_paths = []
        for item in items:
            fallback = f"card_{item['source'].stem}_{item['index']:02d}"
            label = safe_card_label(item["label_widget"].value, fallback)
            output_path = next_available_path(output_dir / f"{label}.png")
            crop_bgr = cv2.cvtColor(item["crop_rgb"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), crop_bgr)
            saved_paths.append(output_path)

        with save_output:
            save_output.clear_output()
            print(f"Saved {len(saved_paths)} cards to {output_dir}")
            for path in saved_paths:
                print(path.name)

    save_button = widgets.Button(description="Save labeled cards", button_style="success", icon="save")
    save_button.on_click(save_cards)

    if not items:
        return widgets.HTML("No reference cards detected.")

    return widgets.VBox([widgets.VBox([reference_image_group(group) for group in groups]), save_button, save_output])


def display_reference_card_labeler(*args, **kwargs) -> None:
    display(create_reference_card_labeler(*args, **kwargs))


def extract_reference_card_groups(
    paths: CardDetectionPaths,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask,
    scale: float,
    colors: tuple[str, ...],
) -> list[dict]:
    groups = []
    for reference_path in sorted((paths.dataset_dir / "reference_images").glob("*.jpg")):
        ref_bgr, ref_rgb = load_image(reference_path)
        result = detect_cards_in_image(
            image_bgr=ref_bgr,
            image_rgb=ref_rgb,
            color_thresholds=color_thresholds,
            rectangle_settings=rectangle_settings,
            card_template_mask=card_template_mask,
            colors=colors,
            scale=scale,
            capture_debug=False,
            detect_special=False,
        )
        items = []
        for index, (title, crop_rgb) in enumerate(result.cards, start=1):
            color = title.split()[1] if len(title.split()) > 1 else "card"
            items.append(
                {
                    "source": reference_path,
                    "index": index,
                    "title": title,
                    "crop_rgb": crop_rgb,
                    "label_widget": widgets.Text(
                        value=f"{color}_{reference_path.stem}_{index:02d}",
                        description="Label",
                        layout=widgets.Layout(width="360px"),
                    ),
                }
            )
        groups.append({"source": reference_path, "overlay": result.overlay, "items": items})
    return groups


def extract_reference_card_items(*args, **kwargs) -> list[dict]:
    return [item for group in extract_reference_card_groups(*args, **kwargs) for item in group["items"]]


def reference_image_group(group: dict):
    overlay_widget = widgets.Image(
        value=cv2.imencode(".png", cv2.cvtColor(group["overlay"], cv2.COLOR_RGB2BGR))[1].tobytes(),
        format="png",
        width=520,
    )
    return widgets.VBox(
        [
            widgets.HTML(f"<h4>{group['source'].name}</h4>"),
            overlay_widget,
            widgets.VBox([reference_card_row(item) for item in group["items"]]),
        ],
        layout=widgets.Layout(margin="0 0 18px 0"),
    )


def reference_card_row(item: dict):
    image_widget = widgets.Image(
        value=cv2.imencode(".png", cv2.cvtColor(item["crop_rgb"], cv2.COLOR_RGB2BGR))[1].tobytes(),
        format="png",
        width=130,
    )
    details = widgets.HTML(f"<b>{item['source'].name}</b><br>{item['title']}")
    return widgets.HBox(
        [image_widget, widgets.VBox([details, item["label_widget"]])],
        layout=widgets.Layout(align_items="center", margin="0 0 10px 0"),
    )


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1
