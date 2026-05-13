import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def extract_hsl_channels(img):
    M, N, C = np.shape(img)
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_l = np.zeros((M, N))

    hlsimg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    data_h = hlsimg[:, :, 0].astype(np.float32) * 2
    data_l = hlsimg[:, :, 1].astype(np.float32) / 255 * 100
    data_s = hlsimg[:, :, 2].astype(np.float32) / 255 * 100

    return data_h, data_s, data_l

def extract_hsv_channels(img):
    M, N, C = np.shape(img)

    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    data_h = hsvimg[:, :, 0].astype(np.float32) * 2
    data_s = hsvimg[:, :, 1].astype(np.float32) / 255 * 100
    data_v = hsvimg[:, :, 2].astype(np.float32) / 255 * 100

    return data_h, data_s, data_v

def plot_colors_histo(img, func, labels):
    channels = func(img=img)
    C2 = len(channels)
    M, N, C1 = img.shape
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, C2)

    mask = np.random.RandomState(seed=0).rand(M, N) < 0.1

    ax = fig.add_subplot(gs[:2, :])
    ax.imshow(img)
    ax.axis('off')

    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])

    ax1.scatter(channels[0][mask].flatten(), channels[1][mask].flatten(), c=img[mask] / 255, s=1, alpha=0.1)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title(f"{labels[0]} vs {labels[1]}")

    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask] / 255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title(f"{labels[0]} vs {labels[2]}")

    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask] / 255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title(f"{labels[1]} vs {labels[2]}")

    plt.tight_layout()
    plt.show()

def mask(
    lower_white_1,
    upper_white_1,
    img_color,
    lower_white_2=None,
    upper_white_2=None,
    lower_white_3=None,
    upper_white_3=None,
    n_masks=3,
    space="hls",
):
    if n_masks not in [1, 2, 3]:
        raise ValueError("n_masks must be 1, 2, or 3")

    space = space.lower()
    img_rgb = np.array(img_color)

    if space == "hls":
        # OpenCV HLS order:
        # H: 0-180, L: 0-255, S: 0-255
        img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS).astype(np.float32)

    elif space == "hsv":
        # OpenCV HSV order:
        # H: 0-180, S: 0-255, V: 0-255
        img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    else:
        raise ValueError("space must be 'hls' or 'hsv'")

    ranges = [
        (lower_white_1, upper_white_1),
        (lower_white_2, upper_white_2),
        (lower_white_3, upper_white_3),
    ]

    masks = []

    for i in range(n_masks):
        lower, upper = ranges[i]

        if lower is None or upper is None:
            raise ValueError(f"Mask {i+1} lower/upper bounds are missing")

        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)

        current_mask = np.all(
            (img_converted >= lower) & (img_converted <= upper),
            axis=2
        )

        masks.append(current_mask)

    combined_mask = masks[0].copy()

    for m in masks[1:]:
        combined_mask |= m

    return masks, combined_mask


def grayscale_mask(img_color, threshold=127, invert=False):
    img_array = np.array(img_color)

    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    if invert:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return mask

def min_area_filter(white_mask_u8, min_area):
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask_u8, connectivity=8)

    clean_mask = np.zeros_like(white_mask_u8, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == label] = 1

    return clean_mask

def black_contours(mask):
    framed_morph_mask = mask.copy()
    framed_morph_mask[0, :] = 0
    framed_morph_mask[-1, :] = 0
    framed_morph_mask[:, 0] = 0
    framed_morph_mask[:, -1] = 0
    return framed_morph_mask

def fill_cards(morph_mask):
    mask = (morph_mask > 0).astype(np.uint8) * 255
    h, w = mask.shape
    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    mask_floodfill = mask.copy()
    cv2.floodFill(mask_floodfill, floodfill_mask, (0, 0), 255)

    holes = cv2.bitwise_not(mask_floodfill)
    filled_mask = ((mask | holes) // 255).astype(np.uint8)
    return filled_mask

def open(kernel_size ,iterations, frame):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel,iterations=iterations)
    return opened_frame

def close(kernel_size ,iterations, frame):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel,iterations=iterations)
    return closed_frame

def erode(kernel_size ,iterations, frame):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_frame = cv2.erode(frame, kernel, iterations=iterations)
    return eroded_frame

def show_cards_on_frame(img_color, edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick_edges = cv2.dilate(edges, kernel, iterations=10)

    img_with_edges = np.array(img_color).copy()
    img_with_edges[thick_edges > 0] = [180, 0, 255]  # Overlay edges in red

    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_edges)
    plt.title("Original Image with Detected Edges")
    plt.axis("off")
    plt.show()

def equalize_hist_rgb(img_color):

    img_array = np.array(img_color)

  
    img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)

    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

    img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)

    return img_equalized


def remove_side_white(white_mask, border_width=1):
    _, labels = cv2.connectedComponents(white_mask.astype(np.uint8))

    bw = border_width
    border_labels = np.unique(np.r_[
        labels[:bw, :].ravel(),
        labels[-bw:, :].ravel(),
        labels[:, :bw].ravel(),
        labels[:, -bw:].ravel()
    ])

    return white_mask & ~np.isin(labels, border_labels)

def dark_frames_segmentation_pipeline(img_color):
    lower_white_1 = np.array([15.0, 0.0, 60.0])
    upper_white_1 = np.array([40.0, 100.0, 100.0])

    lower_white_2 = np.array([10, 0.0, 90.0])
    upper_white_2 = np.array([50, 100.0, 100.0])

    lower_white_3 = np.array([230, 0.0, 80.0])
    upper_white_3 = np.array([330, 100.0, 100.0])

    masks, white_mask = mask(lower_white_1, upper_white_1, img_color,lower_white_2, upper_white_2, lower_white_3, upper_white_3, n_masks=3)
    white_mask_u8 = white_mask.astype(np.uint8) * 255
    closed_frame = close(3,5, white_mask_u8)
    clean_mask = min_area_filter(closed_frame, min_area=10000)
    framed_morph_mask = black_contours(clean_mask)
    eroded=open(5, 1, framed_morph_mask)
    filled_mask = fill_cards(eroded)
    opened_mask = open(5,15,filled_mask)
    clean_filled_mask = min_area_filter(opened_mask, min_area=100000)
    edges = cv2.Canny(clean_filled_mask * 255, 0, 0)
    return  masks,edges

def white_frames_segmentation_pipeline(img_color):
    lower_white_1 = np.array([10, 5, 5])
    upper_white_1 = np.array([360, 100, 100])

    enhanced_img = enhance_image(img_color)
    masks, white_mask = mask(lower_white_1, upper_white_1, enhanced_img,n_masks=1)
    framed_morph_mask = black_contours(white_mask)
    filled_mask = fill_cards(framed_morph_mask)
    open_frame = open(5,25, filled_mask)
    clean_mask = min_area_filter(open_frame, min_area=80000)
    white_mask_no_border=remove_side_white(clean_mask, border_width=20)
    edges = cv2.Canny(white_mask_no_border * 255, 0, 0)
    show_cards_on_frame(img_color, edges)
    return  masks,edges



def save_inside_edges(img_color, edges, image_name, output_dir="extracted_images"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    img_array = np.array(img_color)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    saved_paths = []

    for i, contour in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)

        # Crop original image around contour
        crop = img_array[y:y+h, x:x+w]

        # Shift contour coordinates to crop coordinates
        contour_shifted = contour - [x, y]

        # Create mask for exact contour interior
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)

        # Add alpha channel using the mask
        crop_rgba = np.dstack([crop, mask])

        save_path = output_dir / f"{image_name}_{i}.png"
        Image.fromarray(crop_rgba).save(save_path)

        saved_paths.append(save_path)

    return saved_paths

def enhance_image(img_color):
    img = np.array(img_color).copy()

    denoised = cv2.medianBlur(img, 5)

    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=10.0,
        tileGridSize=(8, 8)
    )
    L_enhanced = clahe.apply(L)

    lab_enhanced = cv2.merge([L_enhanced, A, B])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    return sharpened


