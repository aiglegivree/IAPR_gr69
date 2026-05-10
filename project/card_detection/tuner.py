from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from .io import save_gray_threshold, save_hsv_thresholds, save_threshold_rectangle_settings
from .masking import close_mask, filter_small_components, threshold_gray_image, threshold_hsv_image
from .matcher import fit_card_mask_candidates


def create_threshold_tuner(
    *,
    image_bgr: np.ndarray,
    image_rgb: np.ndarray,
    color: str,
    hsv_ranges: list[tuple[np.ndarray, np.ndarray]],
    gray_settings: dict[str, int],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    hsv_path,
    gray_path,
    rectangle_settings_path,
    use_gray_threshold: bool = False,
    scale: float = 0.25,
):
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError("ipywidgets is required for the tuner") from exc

    clear_output(wait=True)

    hsv_controls = []
    if not use_gray_threshold:
        for index, (lower, upper) in enumerate(hsv_ranges, start=1):
            h_slider = widgets.IntRangeSlider(value=(int(lower[0]), int(upper[0])), min=0, max=179, step=1, description=f"H{index}")
            s_slider = widgets.IntRangeSlider(value=(int(lower[1]), int(upper[1])), min=0, max=255, step=1, description=f"S{index}")
            v_slider = widgets.IntRangeSlider(value=(int(lower[2]), int(upper[2])), min=0, max=255, step=1, description=f"V{index}")
            hsv_controls.append((h_slider, s_slider, v_slider))

    gray_slider = widgets.IntRangeSlider(value=(gray_settings["low"], gray_settings["high"]), min=0, max=255, step=1, description="Gray")
    min_area_slider = widgets.IntSlider(value=rectangle_settings["min_area"], min=0, max=50000, step=50, description="Min area")
    close_size_slider = widgets.IntSlider(value=rectangle_settings["close_size"], min=0, max=51, step=2, description="Close size")
    close_iter_slider = widgets.IntSlider(value=rectangle_settings["close_iter"], min=0, max=50, step=1, description="Close iter")
    min_fill_slider = widgets.IntSlider(value=rectangle_settings["min_fill"], min=1, max=100, step=1, description="Min fill")
    min_score_slider = widgets.IntSlider(value=rectangle_settings["min_score"], min=0, max=10000, step=50, description="Min score")
    angle_step_slider = widgets.IntSlider(value=rectangle_settings["angle_step"], min=2, max=30, step=1, description="Angle step")
    max_outside_slider = widgets.IntSlider(value=rectangle_settings.get("max_outside", 100), min=0, max=100, step=1, description="Max outside")
    save_threshold_button = widgets.Button(description="Save Gray" if use_gray_threshold else "Save HSV", button_style="success")
    save_rectangle_button = widgets.Button(description="Save Rectangles", button_style="success")
    status = widgets.Output()
    preview = widgets.Output()

    def current_threshold_ranges() -> list[tuple[np.ndarray, np.ndarray]]:
        ranges = []
        for h_slider, s_slider, v_slider in hsv_controls:
            h_low, h_high = h_slider.value
            s_low, s_high = s_slider.value
            v_low, v_high = v_slider.value
            ranges.append((np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8)))
        return ranges

    def current_gray_threshold() -> dict[str, int]:
        low, high = sorted(gray_slider.value)
        return {"low": int(low), "high": int(high)}

    def current_rectangle_settings() -> dict[str, int]:
        return {
            "min_area": int(min_area_slider.value),
            "close_size": int(close_size_slider.value),
            "close_iter": int(close_iter_slider.value),
            "min_fill": int(min_fill_slider.value),
            "min_score": int(min_score_slider.value),
            "angle_step": int(angle_step_slider.value),
            "max_outside": int(max_outside_slider.value),
        }

    def current_raw_mask() -> np.ndarray:
        if use_gray_threshold:
            settings = current_gray_threshold()
            return threshold_gray_image(image_bgr, settings["low"], settings["high"])
        return threshold_hsv_image(image_bgr, current_threshold_ranges())

    def update_preview(*_):
        settings = current_rectangle_settings()
        raw_mask = current_raw_mask()
        area_mask = filter_small_components(raw_mask, settings["min_area"])
        processed_mask = close_mask(area_mask, settings["close_size"], settings["close_iter"])
        small_mask = cv2.resize(processed_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        overlay = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()
        all_candidates_overlay = overlay.copy()
        candidates, remaining_mask, all_taken_candidates = fit_card_mask_candidates(small_mask, settings, card_template_mask, return_debug=True)

        for rank, (_, box, (x, y, _, _), _, _, _, outside_ratio) in enumerate(all_taken_candidates, start=1):
            box_int = box.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(all_candidates_overlay, [box_int], True, (255, 165, 0), 4)
            cv2.putText(all_candidates_overlay, f"{rank} out={outside_ratio:.2f}", (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2, cv2.LINE_AA)

        for rank, (_, box, (x, y, _, _), _, _, _, outside_ratio) in enumerate(candidates, start=1):
            box_int = box.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [box_int], True, (255, 0, 0), 4)
            cv2.putText(overlay, f"#{rank} out={outside_ratio:.2f}", (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

        preview.clear_output(wait=True)
        with preview:
            fig, axes = plt.subplots(1, 5, figsize=(24, 7))
            axes[0].imshow(raw_mask, cmap="gray")
            axes[0].set_title(f"{color} {'gray' if use_gray_threshold else 'HSV'} threshold")
            axes[1].imshow(small_mask, cmap="gray")
            axes[1].set_title("Area filtered + closed")
            axes[2].imshow(remaining_mask, cmap="gray")
            axes[2].set_title("Remaining after matches")
            axes[3].imshow(all_candidates_overlay)
            axes[3].set_title("All candidates taken")
            axes[4].imshow(overlay)
            axes[4].set_title("Confirmed detections")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        status.clear_output(wait=True)
        with status:
            print(f"Min area before closing: {settings['min_area']}")
            print(f"Max outside rectangle: {settings['max_outside']}%")
            print(f"Mask pixels: {int(np.count_nonzero(small_mask))}")
            print(f"All candidates taken: {len(all_taken_candidates)}")
            print(f"Confirmed detections: {len(candidates)}")
            for rank, (score, _, (x, y, w, h), (rect_width, rect_height), fill_ratio, _, outside_ratio) in enumerate(candidates, start=1):
                print(
                    f"#{rank}: score={score:.2f}, fill={fill_ratio:.2f}, outside={outside_ratio:.2f}, "
                    f"rect=(width={rect_width:.1f}, height={rect_height:.1f}), bbox=(x={x}, y={y}, w={w}, h={h})"
                )

    def save_threshold_settings(_):
        if use_gray_threshold:
            settings = current_gray_threshold()
            save_gray_threshold(gray_path, settings)
        else:
            save_hsv_thresholds(hsv_path, current_threshold_ranges())
        update_preview()

    def save_rectangle_settings(_):
        save_threshold_rectangle_settings(rectangle_settings_path, current_rectangle_settings())
        update_preview()

    if use_gray_threshold:
        gray_slider.observe(update_preview, names="value")
    else:
        for h_slider, s_slider, v_slider in hsv_controls:
            for widget in (h_slider, s_slider, v_slider):
                widget.observe(update_preview, names="value")
    for widget in (min_area_slider, close_size_slider, close_iter_slider, min_fill_slider, min_score_slider, angle_step_slider, max_outside_slider):
        widget.observe(update_preview, names="value")

    save_threshold_button.on_click(save_threshold_settings)
    save_rectangle_button.on_click(save_rectangle_settings)
    threshold_rows = [widgets.HBox([gray_slider])] if use_gray_threshold else [widgets.HBox([h_slider, s_slider, v_slider]) for h_slider, s_slider, v_slider in hsv_controls]
    widget = widgets.VBox([
        widgets.HTML(f"<b>{color} {'grayscale' if use_gray_threshold else 'HSV'} threshold</b>"),
        *threshold_rows,
        widgets.HBox([save_threshold_button]),
        widgets.HTML("<b>Rectangle mask cleanup</b>"),
        widgets.HBox([min_area_slider, close_size_slider, close_iter_slider]),
        widgets.HBox([min_fill_slider, min_score_slider, angle_step_slider]),
        widgets.HBox([max_outside_slider]),
        widgets.HBox([save_rectangle_button]),
        status,
        preview,
    ])
    display(widget)
    update_preview()
    return widget
