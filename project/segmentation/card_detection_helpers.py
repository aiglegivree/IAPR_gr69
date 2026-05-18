import cv2
import numpy as np


def min_area_filter(white_mask_u8, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_mask_u8,
        connectivity=8,
    )

    clean_mask = np.zeros_like(white_mask_u8, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == label] = 1

    return clean_mask


def open(kernel_size, iterations, frame):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=iterations)


def close(kernel_size, iterations, frame):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def find_circle_hough(mask, expected_radius, radius_tolerance=5, min_score=0.5):
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

    for x, y, r in circles:
        circle_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.circle(circle_mask, (int(x), int(y)), int(r), 1, -1)

        circle_area = np.sum(circle_mask)
        if circle_area == 0:
            continue

        foreground_inside = np.sum(binary_mask & circle_mask)
        fill_score = foreground_inside / circle_area
        radius_error = abs(r - expected_radius) / expected_radius
        score = fill_score - radius_error

        if score > best_score:
            best_score = score
            best_circle = {
                "center": (int(x), int(y)),
                "radius": int(r),
                "score": float(score),
                "fill_score": float(fill_score),
                "radius_error": float(radius_error),
            }

    if best_circle is None or best_circle["score"] < min_score:
        return None

    return best_circle


def mask_center(mask):
    mask = (mask > 0).astype(np.uint8)
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


def biggest_object(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest_label = 1 + np.argmax(areas)
    return (labels == biggest_label).astype(np.uint8)


def keep_objects_surrounded_by_white(mask, white_mask, min_white_ratio, ring_radius):
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


def get_holes(mask, min_hole_area=1):
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
