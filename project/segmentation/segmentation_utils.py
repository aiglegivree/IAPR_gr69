from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image


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
    """Select the fastest available OpenCV backend for mask operations."""
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
    """Set the global OpenCV backend used by thresholding functions."""
    global ACCELERATION_BACKEND
    ACCELERATION_BACKEND = backend


def load_hsv_ranges(color):
    """Load one or more HSV threshold ranges for a named UNO color."""
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
    """Convert PIL or NumPy image input into an RGB image representation."""
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


def min_area_filter(mask, min_area):
    """Keep connected components whose area is at least the requested size."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == label] = 1

    return clean_mask


def morph_open(mask, kernel_size=3, iterations=1):
    """Apply morphological opening to remove small foreground noise."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(mask, kernel_size=3, iterations=1):
    """Apply morphological closing to fill small foreground gaps."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def color_mask_from_threshold_file(img_color, color):
    """Threshold an RGB image into a binary mask for one configured color."""
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
    """Build cleaned binary masks for every configured UNO color."""
    color_masks = {
        color: color_mask_from_threshold_file(img_color, color)
        for color in COLORS
    }

    for color, mask in color_masks.items():
        closed_mask = morph_close(mask, kernel_size=3, iterations=1)
        color_masks[color] = min_area_filter(closed_mask, MIN_OBJECT_AREA)

    return color_masks


def image_token_type(img_color):
    """Classify the center token as yellow or grey from image brightness."""
    grey_mean = float(np.array(img_color, dtype=np.float32).mean())
    if grey_mean < 185:
        return "yellow"
    return "grey"


def find_circle_hough(mask, expected_radius, radius_tolerance=5, min_score=0.5):
    """Find the best circular component near an expected radius in a mask."""
    binary_mask = (mask > 0).astype(np.uint8)
    img = cv2.medianBlur(binary_mask * 255, 5)

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=expected_radius * 2,
        param1=100,
        param2=15,
        minRadius=int(expected_radius - radius_tolerance),
        maxRadius=int(expected_radius + radius_tolerance),
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best_circle = None
    best_score = -np.inf

    for x, y, radius in circles:
        circle_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.circle(circle_mask, (int(x), int(y)), int(radius), 1, -1)

        circle_area = np.sum(circle_mask)
        if circle_area == 0:
            continue

        foreground_inside = np.sum(binary_mask & circle_mask)
        fill_score = foreground_inside / circle_area
        radius_error = abs(radius - expected_radius) / expected_radius
        score = fill_score - radius_error

        if score > best_score:
            best_score = score
            best_circle = {
                "center": (int(x), int(y)),
                "radius": int(radius),
                "score": float(score),
                "fill_score": float(fill_score),
                "radius_error": float(radius_error),
            }

    if best_circle is None or best_circle["score"] < min_score:
        return None

    return best_circle


def mask_center(mask):
    """Return the centroid of foreground pixels in a binary mask."""
    mask = (mask > 0).astype(np.uint8)
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


def biggest_object(mask):
    """Return a binary mask containing only the largest connected component."""
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest_label = 1 + np.argmax(areas)
    return (labels == biggest_label).astype(np.uint8)


def get_token_center(image):
    """Locate the center of the yellow or grey center token in an image."""
    img_color = normalize_image_input(image)
    token = image_token_type(img_color)

    if token == "yellow":
        yellow_mask = color_mask_from_threshold_file(img_color, "yellow")
        yellow_mask = morph_close(yellow_mask, kernel_size=3, iterations=1)
        yellow_mask = min_area_filter(yellow_mask, MIN_OBJECT_AREA)
        coords = find_circle_hough(
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
    token_mask = biggest_object(grey_mask)

    if token_mask is None:
        return None

    return mask_center(token_mask)


def remove_token_from_color_masks(img_color, color_masks):
    """Remove the center token from color masks before card processing."""
    token = image_token_type(img_color)

    if token == "yellow":
        center = get_token_center(img_color)
        if center is None:
            return color_masks

        token_mask = np.ones_like(color_masks["yellow"], dtype=np.uint8)
        x, y = center
        coords = find_circle_hough(
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
    token_mask = biggest_object(grey_mask)

    if token_mask is None:
        return color_masks

    color_masks["black"] = color_masks["black"] & ~token_mask
    color_masks["black"] = min_area_filter(color_masks["black"], MIN_OBJECT_AREA)
    return color_masks


def keep_objects_surrounded_by_white(mask, white_mask, min_white_ratio, ring_radius):
    """Keep connected components that have enough white pixels around them."""
    object_mask = (mask > 0).astype(np.uint8)
    white_mask = (white_mask > 0).astype(np.uint8)
    kept_mask = np.zeros_like(object_mask, dtype=np.uint8)
    object_scores = []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        object_mask,
        connectivity=8,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )

    height, width = object_mask.shape
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        left = max(0, x - ring_radius)
        top = max(0, y - ring_radius)
        right = min(width, x + w + ring_radius)
        bottom = min(height, y + h + ring_radius)

        label_roi = labels[top:bottom, left:right]
        current_object = (label_roi == label).astype(np.uint8)
        surrounding_area = cv2.dilate(current_object, kernel, iterations=1)
        surrounding_ring = (surrounding_area > 0) & (current_object == 0)
        ring_area = int(surrounding_ring.sum())

        if ring_area == 0:
            white_ratio = 0.0
        else:
            white_roi = white_mask[top:bottom, left:right]
            white_ratio = float((white_roi[surrounding_ring] > 0).sum() / ring_area)

        object_scores.append({
            "label": label,
            "area": int(stats[label, cv2.CC_STAT_AREA]),
            "white_ratio": white_ratio,
            "kept": white_ratio >= min_white_ratio,
        })

        if white_ratio >= min_white_ratio:
            kept_mask_roi = kept_mask[top:bottom, left:right]
            kept_mask_roi[current_object > 0] = 1

    return kept_mask, object_scores


def keep_white_surrounded_objects(color_masks):
    """Filter each non-white color mask to card-like objects with white borders."""
    for color in COLORS:
        if color == "white":
            continue

        filtered_mask, _ = keep_objects_surrounded_by_white(
            color_masks[color],
            color_masks["white"],
            WHITE_SURROUND_THRESHOLD,
            SURROUND_RING_RADIUS,
        )
        color_masks[color] = morph_close(filtered_mask, kernel_size=5, iterations=1)

    return color_masks


def get_holes(mask, min_hole_area=1):
    """Find background components fully enclosed by foreground shapes."""
    fg = (mask > 0).astype(np.uint8)
    _, shape_labels = cv2.connectedComponents(fg, connectivity=8)

    bg = 1 - fg
    num_bg, bg_labels = cv2.connectedComponents(bg, connectivity=8)
    hole_mask = np.zeros_like(fg, dtype=np.uint8)

    for bg_label in range(1, num_bg):
        component = bg_labels == bg_label

        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if touches_border:
            continue

        if np.count_nonzero(component) < min_hole_area:
            continue

        component_uint8 = component.astype(np.uint8)
        dilated = cv2.dilate(
            component_uint8,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        neighboring_shape_labels = np.unique(shape_labels[dilated.astype(bool)])
        neighboring_shape_labels = neighboring_shape_labels[neighboring_shape_labels != 0]

        if len(neighboring_shape_labels) == 0:
            continue

        hole_mask[component] = 1

    return hole_mask


def filter_shapes_by_dilated_ring_overlap_binary(
    mask,
    white_mask,
    dilation_radius=5,
    min_white_pixels=10,
):
    """Keep mask components whose dilated ring overlaps enough white pixels."""
    binary = (mask > 0).astype(np.uint8)
    white_binary = white_mask > 0

    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    filtered_mask = np.zeros_like(binary, dtype=np.uint8)

    ksize = 2 * dilation_radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    h, w = binary.shape

    for label_id in range(1, num_labels):
        x = stats_cc[label_id, cv2.CC_STAT_LEFT]
        y = stats_cc[label_id, cv2.CC_STAT_TOP]
        bw = stats_cc[label_id, cv2.CC_STAT_WIDTH]
        bh = stats_cc[label_id, cv2.CC_STAT_HEIGHT]

        x0 = max(x - dilation_radius, 0)
        y0 = max(y - dilation_radius, 0)
        x1 = min(x + bw + dilation_radius, w)
        y1 = min(y + bh + dilation_radius, h)

        local_shape = (labels[y0:y1, x0:x1] == label_id).astype(np.uint8)
        local_dilated = cv2.dilate(local_shape, kernel, iterations=1)
        dilation_ring = (local_dilated == 1) & (local_shape == 0)
        local_white = white_binary[y0:y1, x0:x1]

        white_pixels_in_ring = np.count_nonzero(dilation_ring & local_white)
        if white_pixels_in_ring >= min_white_pixels:
            filtered_mask[labels == label_id] = 255

    return filtered_mask


def extract_symbols_mask_per_color(img_color, color_masks):
    """Extract cleaned symbol masks from the holes of each colored card mask."""
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

        hole_mask = get_holes(mask, min_hole_area=100)
        hole_mask_filtered = filter_shapes_by_dilated_ring_overlap_binary(
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
        hole_mask_filtered = morph_open(hole_mask_filtered, kernel_size=3, iterations=1)

        symbols_mask_per_color[color] = hole_mask_filtered

    return symbols_mask_per_color


def get_symbols_mask_per_color(img_color):
    """Run the full mask pipeline and return final symbol masks per card color."""
    color_masks = get_color_masks(img_color)
    color_masks = remove_token_from_color_masks(img_color, color_masks)
    color_masks = keep_white_surrounded_objects(color_masks)
    return extract_symbols_mask_per_color(img_color, color_masks)


def crop_centered_binary_mask(mask, center_xy, patch_size=150):
    """Crop a fixed-size binary patch centered on a symbol location."""
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
    """Merge nearby symbol pixels and return patches with color and center."""
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
    """Detect UNO card symbols and return binary patch, color, center records."""
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
