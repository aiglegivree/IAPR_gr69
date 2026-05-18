from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image

import card_detection_helpers as helpers


BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = BASE_DIR / "iapr-26-uno-vision-challenge" / "train_images"
TEST_DIR = BASE_DIR / "iapr-26-uno-vision-challenge" / "test_images"
HSV_DIR = BASE_DIR / "hsv_thresholds"

COLORS = ["blue", "green", "red", "yellow", "black", "white"]
MIN_OBJECT_AREA = 5000
WHITE_SURROUND_THRESHOLD = 0.5
SURROUND_RING_RADIUS = 20
SYMBOL_PATCH_SIZE = 150
SYMBOL_MERGE_DISTANCE = 40
ACCELERATION_BACKEND = "cpu"


def configure_acceleration():
    try:
        cuda_ready = (
            hasattr(cv2, "cuda")
            and cv2.cuda.getCudaEnabledDeviceCount() > 0
            and hasattr(cv2, "cuda_GpuMat")
            and hasattr(cv2.cuda, "cvtColor")
            and hasattr(cv2.cuda, "inRange")
            and hasattr(cv2.cuda, "bitwise_or")
            and hasattr(cv2.cuda, "bitwise_and")
        )
    except cv2.error:
        cuda_ready = False

    if cuda_ready:
        return "cuda"

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            return "opencl"

    return "cpu"


def set_acceleration_backend(backend):
    global ACCELERATION_BACKEND
    ACCELERATION_BACKEND = backend


def load_hsv_ranges(color):
    threshold_path = HSV_DIR / f"{color}_hsv.txt"
    text = threshold_path.read_text()
    ranges = []
    idx = 1

    while True:
        lower_match = re.search(rf"lower_hsv_{idx}\s*=\s*np\.array\(\[([^\]]+)\]\)", text)
        upper_match = re.search(rf"upper_hsv_{idx}\s*=\s*np\.array\(\[([^\]]+)\]\)", text)

        if lower_match is None or upper_match is None:
            break

        lower = np.array(
            [int(value.strip()) for value in lower_match.group(1).split(",")],
            dtype=np.uint8,
        )
        upper = np.array(
            [int(value.strip()) for value in upper_match.group(1).split(",")],
            dtype=np.uint8,
        )
        ranges.append((lower, upper))
        idx += 1

    if not ranges:
        raise ValueError(f"No HSV ranges found in {threshold_path}")

    return ranges


def normalize_image_input(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    image_array = np.asarray(image)

    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = image_array * 255
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    return image_array


def color_mask_from_threshold_file(img_color, color):
    img_array = np.array(img_color)

    if ACCELERATION_BACKEND == "cuda":
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_array)
        hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2HSV)
        combined_mask = cv2.cuda_GpuMat()
        combined_mask.upload(np.zeros(img_array.shape[:2], dtype=np.uint8))

        for lower, upper in load_hsv_ranges(color):
            current_mask = cv2.cuda.inRange(hsv, lower, upper)
            combined_mask = cv2.cuda.bitwise_or(combined_mask, current_mask)

        return combined_mask.download()

    if ACCELERATION_BACKEND == "opencl":
        hsv = cv2.cvtColor(cv2.UMat(img_array), cv2.COLOR_RGB2HSV)
        combined_mask = cv2.UMat(np.zeros(img_array.shape[:2], dtype=np.uint8))

        for lower, upper in load_hsv_ranges(color):
            current_mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, current_mask)

        return combined_mask.get()

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in load_hsv_ranges(color):
        current_mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, current_mask)

    return combined_mask


def get_color_masks(img_color):
    color_masks = {
        color: color_mask_from_threshold_file(img_color, color)
        for color in COLORS
    }

    for color, mask in color_masks.items():
        closed_mask = helpers.close(3, 1, mask)
        color_masks[color] = helpers.min_area_filter(closed_mask, MIN_OBJECT_AREA)

    return color_masks


def image_token_type(img_color):
    grey_mean = float(np.array(img_color, dtype=np.float32).mean())
    if grey_mean < 185:
        return "yellow"
    return "grey"


def get_token_center(image):
    img_color = normalize_image_input(image)
    token = image_token_type(img_color)

    if token == "yellow":
        yellow_mask = color_mask_from_threshold_file(img_color, "yellow")
        yellow_mask = helpers.close(3, 1, yellow_mask)
        yellow_mask = helpers.min_area_filter(yellow_mask, MIN_OBJECT_AREA)
        coords = helpers.find_circle_hough(
            mask=yellow_mask,
            expected_radius=100,
            radius_tolerance=50,
        )

        if coords is None:
            return None

        return coords["center"]

    lower, upper = load_hsv_ranges("grey")[0]
    hsv = cv2.cvtColor(np.array(img_color), cv2.COLOR_RGB2HSV)
    grey_mask = cv2.inRange(hsv, lower, upper)
    token_mask = helpers.biggest_object(grey_mask)

    if token_mask is None:
        return None

    return helpers.mask_center(token_mask)


def remove_token_from_color_masks(img_color, color_masks):
    token = image_token_type(img_color)

    if token == "yellow":
        center = get_token_center(img_color)
        if center is None:
            return color_masks

        token_mask = np.ones_like(color_masks["yellow"], dtype=np.uint8)
        x, y = center
        coords = helpers.find_circle_hough(
            mask=color_masks["yellow"],
            expected_radius=100,
            radius_tolerance=50,
        )
        radius = coords["radius"] * 1.2 if coords is not None else 120
        cv2.circle(token_mask, (int(x), int(y)), int(radius), 0, -1)
        color_masks["yellow"] = cv2.bitwise_and(
            color_masks["yellow"],
            color_masks["yellow"],
            mask=token_mask,
        )
        return color_masks

    lower, upper = load_hsv_ranges("grey")[0]
    hsv = cv2.cvtColor(np.array(img_color), cv2.COLOR_RGB2HSV)
    grey_mask = cv2.inRange(hsv, lower, upper)
    token_mask = helpers.biggest_object(grey_mask)

    if token_mask is None:
        return color_masks

    color_masks["black"] = color_masks["black"] & ~token_mask
    color_masks["black"] = helpers.min_area_filter(color_masks["black"], MIN_OBJECT_AREA)
    return color_masks


def keep_white_surrounded_objects(color_masks):
    for color in COLORS:
        if color == "white":
            continue

        filtered_mask, _ = helpers.keep_objects_surrounded_by_white(
            color_masks[color],
            color_masks["white"],
            WHITE_SURROUND_THRESHOLD,
            SURROUND_RING_RADIUS,
        )
        color_masks[color] = helpers.close(5, 1, filtered_mask)

    return color_masks


def extract_symbols_mask_per_color(img_color, color_masks):
    symbols_mask_per_color = {}
    img_array = np.array(img_color)

    if ACCELERATION_BACKEND == "cuda":
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_array)
        hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2HSV)
        white_by_hsv = cv2.cuda.inRange(
            hsv,
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([179, 40, 255], dtype=np.uint8),
        ).download()
    elif ACCELERATION_BACKEND == "opencl":
        hsv = cv2.cvtColor(cv2.UMat(img_array), cv2.COLOR_RGB2HSV)
        white_by_hsv = cv2.inRange(
            hsv,
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([179, 40, 255], dtype=np.uint8),
        ).get()
    else:
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        white_by_hsv = (hsv[:, :, 1] <= 40).astype(np.uint8)

    for color, mask in color_masks.items():
        if color == "white":
            continue

        hole_mask = helpers.get_holes(mask, min_hole_area=100)
        hole_mask_filtered = helpers.filter_shapes_by_dilated_ring_overlap_binary(
            hole_mask,
            color_masks["white"],
            dilation_radius=20,
            min_white_pixels=50,
        )
        hole_mask_filtered = cv2.dilate(
            hole_mask_filtered,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        hole_mask_filtered = cv2.bitwise_and(hole_mask_filtered, white_by_hsv)
        hole_mask_filtered = helpers.open(3, 1, hole_mask_filtered)

        symbols_mask_per_color[color] = hole_mask_filtered

    return symbols_mask_per_color


def get_symbols_mask_per_color(img_color):
    color_masks = get_color_masks(img_color)
    color_masks = remove_token_from_color_masks(img_color, color_masks)
    color_masks = keep_white_surrounded_objects(color_masks)
    return extract_symbols_mask_per_color(img_color, color_masks)


def crop_centered_binary_mask(mask, center_xy, patch_size=150):
    binary_mask = (mask > 0).astype(np.uint8) * 255
    h, w = binary_mask.shape
    cx, cy = center_xy

    left = int(round(cx)) - patch_size // 2
    top = int(round(cy)) - patch_size // 2
    right = left + patch_size
    bottom = top + patch_size

    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, w)
    src_bottom = min(bottom, h)

    dst_left = src_left - left
    dst_top = src_top - top

    patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
    patch[
        dst_top:dst_top + (src_bottom - src_top),
        dst_left:dst_left + (src_right - src_left),
    ] = binary_mask[src_top:src_bottom, src_left:src_right]

    return patch


def extract_merged_symbol_records(mask, color, merge_distance=30, patch_size=150):
    binary_mask = (mask > 0).astype(np.uint8)

    if not np.any(binary_mask):
        return []

    merge_kernel_size = 2 * int(np.ceil(merge_distance / 2)) + 1
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (merge_kernel_size, merge_kernel_size),
    )
    merged_mask = cv2.dilate(binary_mask, merge_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        merged_mask,
        connectivity=8,
    )

    records = []
    height, width = binary_mask.shape
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        left = max(0, x - merge_distance)
        top = max(0, y - merge_distance)
        right = min(width, x + w + merge_distance)
        bottom = min(height, y + h + merge_distance)

        label_roi = labels[top:bottom, left:right]
        binary_roi = binary_mask[top:bottom, left:right]
        original_pixels_roi = binary_roi & (label_roi == label).astype(np.uint8)

        if not np.any(original_pixels_roi):
            continue

        ys, xs = np.where(original_pixels_roi > 0)
        center_xy_roi = (float(xs.mean()), float(ys.mean()))
        center_xy = (center_xy_roi[0] + left, center_xy_roi[1] + top)
        symbol_patch = crop_centered_binary_mask(
            original_pixels_roi,
            center_xy_roi,
            patch_size,
        )
        records.append([symbol_patch, color, center_xy])

    return records


def detect_symbols(image):
    img_color = normalize_image_input(image)
    symbols_mask_per_color = get_symbols_mask_per_color(img_color)
    detected_symbols = []

    for color, symbol_mask in symbols_mask_per_color.items():
        detected_symbols.extend(
            extract_merged_symbol_records(
                symbol_mask,
                color,
                merge_distance=SYMBOL_MERGE_DISTANCE,
                patch_size=SYMBOL_PATCH_SIZE,
            )
        )

    return detected_symbols
