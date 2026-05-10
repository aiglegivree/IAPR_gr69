from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from .io import save_gray_threshold, save_hsv_thresholds, save_special_settings, save_threshold_rectangle_settings
from .masking import close_mask, filter_small_components, threshold_gray_image, threshold_hsv_image
from .matcher import fit_card_mask_candidates
from .special import (
    detect_black_rectangle,
    detect_yellow_circle,
    remaining_after_card_fits,
    scaled_color_mask,
)


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
    max_candidates_slider = widgets.IntSlider(value=rectangle_settings.get("max_candidates_per_region", 12), min=1, max=50, step=1, description="Tests/region")
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
            "max_candidates_per_region": int(max_candidates_slider.value),
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
        accepted_ids = {id(candidate) for candidate in candidates}

        for rank, candidate in enumerate(all_taken_candidates, start=1):
            _, box, (x, y, _, _), _, _, _, outside_ratio = candidate
            box_int = box.astype(np.int32).reshape(-1, 1, 2)
            accepted = id(candidate) in accepted_ids
            line_color = (0, 220, 0) if accepted else (255, 80, 80)
            label = f"{rank} ok" if accepted else f"{rank} reject"
            cv2.polylines(all_candidates_overlay, [box_int], True, line_color, 4)
            cv2.putText(all_candidates_overlay, f"{label} out={outside_ratio:.2f}", (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)

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
            print(f"Tests per region: {settings['max_candidates_per_region']}")
            print(f"Mask pixels: {int(np.count_nonzero(small_mask))}")
            print(f"All candidates taken: {len(all_taken_candidates)}")
            print(f"Confirmed detections: {len(candidates)}")
            for rank, (score, _, (x, y, w, h), (rect_width, rect_height), fill_ratio, _, outside_ratio) in enumerate(candidates, start=1):
                print(
                    f"Accepted #{rank}: score={score:.2f}, fill={fill_ratio:.2f}, outside={outside_ratio:.2f}, "
                    f"rect=(width={rect_width:.1f}, height={rect_height:.1f}), bbox=(x={x}, y={y}, w={w}, h={h})"
                )
            rejected_candidates = [candidate for candidate in all_taken_candidates if id(candidate) not in accepted_ids]
            for rank, (score, _, (x, y, w, h), (rect_width, rect_height), fill_ratio, _, outside_ratio) in enumerate(rejected_candidates, start=1):
                print(
                    f"Rejected #{rank}: {candidate_rejection_reason(score, fill_ratio, outside_ratio, settings)}; "
                    f"score={score:.2f}, fill={fill_ratio:.2f}, outside={outside_ratio:.2f}, "
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
    for widget in (min_area_slider, close_size_slider, close_iter_slider, min_fill_slider, min_score_slider, angle_step_slider, max_outside_slider, max_candidates_slider):
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
        widgets.HBox([max_outside_slider, max_candidates_slider]),
        widgets.HBox([save_rectangle_button]),
        status,
        preview,
    ])
    display(widget)
    update_preview()
    return widget


def candidate_rejection_reason(score: float, fill_ratio: float, outside_ratio: float, settings: dict[str, int]) -> str:
    reasons = []
    min_score = float(settings.get("min_score", 0))
    min_fill = settings["min_fill"] / 100.0
    max_outside = settings.get("max_outside", 100) / 100.0
    if score < min_score:
        reasons.append(f"score {score:.2f} < min_score {min_score:.2f}")
    if fill_ratio < min_fill:
        reasons.append(f"fill {fill_ratio:.2f} < min_fill {min_fill:.2f}")
    if outside_ratio > max_outside:
        reasons.append(f"outside {outside_ratio:.2f} > max_outside {max_outside:.2f}")
    return "; ".join(reasons) if reasons else "suppressed by a later filter"


def create_special_tuner(
    *,
    image_bgr: np.ndarray,
    image_rgb: np.ndarray,
    color_thresholds: dict[str, dict],
    rectangle_settings: dict[str, int],
    card_template_mask: np.ndarray,
    special_settings: dict[str, int],
    special_settings_path,
    yellow_hsv_path=None,
    black_gray_path=None,
    scale: float = 0.25,
):
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError("ipywidgets is required for the tuner") from exc

    clear_output(wait=True)

    yellow_hsv_controls = create_hsv_controls("Y", color_thresholds["yellow"]["ranges"])
    black_gray_settings = color_thresholds["black"].get("settings", {"low": 0, "high": 100})
    black_gray_slider = widgets.IntRangeSlider(value=(black_gray_settings["low"], black_gray_settings["high"]), min=0, max=255, step=1, description="B gray")

    yellow_min_area = widgets.IntSlider(value=special_settings["yellow_min_area"], min=0, max=1000, step=5, description="Y area")
    yellow_min_radius = widgets.IntSlider(value=special_settings["yellow_min_radius"], min=1, max=50, step=1, description="Y radius")
    yellow_min_circularity = widgets.IntSlider(value=special_settings["yellow_min_circularity"], min=0, max=100, step=1, description="Y circle")
    yellow_min_fill = widgets.IntSlider(value=special_settings["yellow_min_fill"], min=0, max=100, step=1, description="Y fill")

    black_min_area = widgets.IntSlider(value=special_settings["black_min_area"], min=0, max=1000, step=5, description="B area")
    black_min_fill = widgets.IntSlider(value=special_settings["black_min_fill"], min=0, max=100, step=1, description="B fill")
    black_max_long_side = widgets.IntSlider(value=special_settings["black_max_long_side"], min=1, max=250, step=1, description="B max long")
    black_min_short_side = widgets.IntSlider(value=special_settings["black_min_short_side"], min=1, max=50, step=1, description="B min short")
    black_max_aspect = widgets.IntSlider(value=special_settings["black_max_aspect"], min=1, max=40, step=1, description="B aspect")
    remove_cards_first = widgets.Checkbox(value=False, description="Preview after removing fitted cards")

    save_button = widgets.Button(description="Save Specials", button_style="success")
    save_yellow_hsv_button = widgets.Button(description="Save Yellow HSV", button_style="success")
    save_black_gray_button = widgets.Button(description="Save Black Gray", button_style="success")
    status = widgets.Output()
    preview = widgets.Output()

    def current_settings() -> dict[str, int]:
        return {
            "yellow_min_area": int(yellow_min_area.value),
            "yellow_min_radius": int(yellow_min_radius.value),
            "yellow_min_circularity": int(yellow_min_circularity.value),
            "yellow_min_fill": int(yellow_min_fill.value),
            "black_min_area": int(black_min_area.value),
            "black_min_fill": int(black_min_fill.value),
            "black_max_long_side": int(black_max_long_side.value),
            "black_min_short_side": int(black_min_short_side.value),
            "black_max_aspect": int(black_max_aspect.value),
        }

    def current_yellow_threshold() -> dict:
        return {"mode": "hsv", "ranges": current_hsv_ranges(yellow_hsv_controls)}

    def current_black_threshold() -> dict:
        low, high = sorted(black_gray_slider.value)
        return {"mode": "gray", "settings": {"low": int(low), "high": int(high)}}

    def special_masks() -> tuple[np.ndarray, np.ndarray]:
        current_thresholds = {
            **color_thresholds,
            "yellow": current_yellow_threshold(),
            "black": current_black_threshold(),
        }
        if remove_cards_first.value:
            yellow_mask = remaining_after_card_fits(image_bgr, "yellow", current_thresholds, rectangle_settings, card_template_mask, scale)
            black_mask = remaining_after_card_fits(image_bgr, "black", current_thresholds, rectangle_settings, card_template_mask, scale)
        else:
            yellow_mask = scaled_color_mask(image_bgr, "yellow", current_thresholds, rectangle_settings, scale)
            black_mask = scaled_color_mask(image_bgr, "black", current_thresholds, rectangle_settings, scale)
        return yellow_mask, black_mask

    def update_preview(*_):
        settings = current_settings()
        yellow_mask, black_mask = special_masks()
        yellow_circle = detect_yellow_circle(yellow_mask, settings)
        black_rectangle = detect_black_rectangle(black_mask, settings)
        overlay = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()

        if yellow_circle is not None:
            cx, cy = yellow_circle["center"]
            radius = yellow_circle["radius"]
            center = (int(round(cx)), int(round(cy)))
            cv2.circle(overlay, center, int(round(radius)), (255, 255, 0), 2)
            cv2.putText(overlay, "yellow", (center[0] + 6, max(20, center[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        if black_rectangle is not None:
            box = black_rectangle["box"].astype(np.int32).reshape(-1, 1, 2)
            x, y, _, _ = cv2.boundingRect(box)
            cv2.polylines(overlay, [box], True, (0, 0, 0), 2)
            cv2.putText(overlay, "black", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        preview.clear_output(wait=True)
        with preview:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            axes[0].imshow(yellow_mask, cmap="gray")
            axes[0].set_title("Yellow token mask")
            axes[1].imshow(black_mask, cmap="gray")
            axes[1].set_title("Black token mask")
            axes[2].imshow(overlay)
            axes[2].set_title("Accepted specials")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        status.clear_output(wait=True)
        with status:
            print("Preview mask:", "remaining after fitted cards" if remove_cards_first.value else "direct color mask")
            if yellow_circle is None:
                print("Yellow circle: none")
            else:
                print(
                    f"Yellow circle: score={yellow_circle['score']:.2f}, fill={yellow_circle['fill']:.2f}, "
                    f"circularity={yellow_circle['circularity']:.2f}, radius={yellow_circle['radius']:.1f}, bbox={yellow_circle['bbox']}"
                )
            if black_rectangle is None:
                print("Black rectangle: none")
            else:
                width, height = black_rectangle["size"]
                long_side = max(width, height)
                short_side = min(width, height)
                aspect = long_side / max(short_side, 1e-6)
                print(
                    f"Black rectangle: score={black_rectangle['score']:.2f}, fill={black_rectangle['fill']:.2f}, "
                    f"long={long_side:.1f}, short={short_side:.1f}, aspect={aspect:.2f}, bbox={black_rectangle['bbox']}"
                )

    def save_settings(_):
        save_special_settings(special_settings_path, current_settings())
        update_preview()

    def save_yellow_hsv(_):
        if yellow_hsv_path is not None:
            save_hsv_thresholds(yellow_hsv_path, current_hsv_ranges(yellow_hsv_controls))
        update_preview()

    def save_black_gray(_):
        if black_gray_path is not None:
            low, high = sorted(black_gray_slider.value)
            save_gray_threshold(black_gray_path, {"low": int(low), "high": int(high)})
        update_preview()

    controls = (
        *flatten_hsv_controls(yellow_hsv_controls),
        black_gray_slider,
        yellow_min_area,
        yellow_min_radius,
        yellow_min_circularity,
        yellow_min_fill,
        black_min_area,
        black_min_fill,
        black_max_long_side,
        black_min_short_side,
        black_max_aspect,
        remove_cards_first,
    )
    for widget_control in controls:
        widget_control.observe(update_preview, names="value")
    save_button.on_click(save_settings)
    save_yellow_hsv_button.on_click(save_yellow_hsv)
    save_black_gray_button.on_click(save_black_gray)

    widget = widgets.VBox([
        widgets.HTML("<b>Yellow special HSV threshold</b>"),
        *hsv_control_rows(yellow_hsv_controls),
        widgets.HBox([save_yellow_hsv_button]),
        widgets.HTML("<b>Black special grayscale threshold</b>"),
        widgets.HBox([black_gray_slider]),
        widgets.HBox([save_black_gray_button]),
        widgets.HTML("<b>Yellow token circle</b>"),
        widgets.HBox([yellow_min_area, yellow_min_radius]),
        widgets.HBox([yellow_min_circularity, yellow_min_fill]),
        widgets.HTML("<b>Black token rectangle</b>"),
        widgets.HBox([black_min_area, black_min_fill]),
        widgets.HBox([black_max_long_side, black_min_short_side, black_max_aspect]),
        widgets.HBox([remove_cards_first, save_button]),
        status,
        preview,
    ])
    display(widget)
    update_preview()
    return widget


def create_hsv_controls(prefix: str, ranges: list[tuple[np.ndarray, np.ndarray]]):
    import ipywidgets as widgets

    controls = []
    for index, (lower, upper) in enumerate(ranges, start=1):
        h_slider = widgets.IntRangeSlider(value=(int(lower[0]), int(upper[0])), min=0, max=179, step=1, description=f"{prefix} H{index}")
        s_slider = widgets.IntRangeSlider(value=(int(lower[1]), int(upper[1])), min=0, max=255, step=1, description=f"{prefix} S{index}")
        v_slider = widgets.IntRangeSlider(value=(int(lower[2]), int(upper[2])), min=0, max=255, step=1, description=f"{prefix} V{index}")
        controls.append((h_slider, s_slider, v_slider))
    return controls


def current_hsv_ranges(hsv_controls):
    ranges = []
    for h_slider, s_slider, v_slider in hsv_controls:
        h_low, h_high = h_slider.value
        s_low, s_high = s_slider.value
        v_low, v_high = v_slider.value
        ranges.append((np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8)))
    return ranges


def flatten_hsv_controls(hsv_controls):
    return [widget for row in hsv_controls for widget in row]


def hsv_control_rows(hsv_controls):
    import ipywidgets as widgets

    return [widgets.HBox([h_slider, s_slider, v_slider]) for h_slider, s_slider, v_slider in hsv_controls]
