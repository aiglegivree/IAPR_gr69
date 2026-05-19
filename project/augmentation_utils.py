"""
augmentation_utils.py
---------------------
Core image-augmentation functions.
No I/O, no plotting, no dataset management — just pixel transforms.
"""

import random
from PIL import Image


def augment_image(img: Image.Image) -> tuple[Image.Image, float, float, float]:
    """
    Apply a random rotation and small random translation to *img*.

    After the geometric transforms the result is re-binarized
    (thresholded at 128) to remove the grey anti-aliasing fringing
    introduced by bilinear interpolation, keeping the same hard
    black/white look as the originals.

    Parameters
    ----------
    img : PIL Image (RGB or L)

    Returns
    -------
    (augmented_image, angle_deg, tx_pixels, ty_pixels)
    """
    angle = random.uniform(0, 360)   # digits can appear in any orientation
    tx    = random.uniform(-10, 10)  # small shift to mimic off-centre capture
    ty    = random.uniform(-10, 10)

    aug = img.rotate(angle, resample=Image.BILINEAR)
    aug = aug.transform(
        img.size,
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.BILINEAR,
    )

    # Re-binarize: bilinear interpolation creates grey anti-aliasing
    # artefacts; thresholding at 128 restores a clean binary image.
    aug_gray = aug.convert("L")
    aug_bin  = aug_gray.point(lambda p: 255 if p >= 128 else 0, "L")
    aug_out  = aug_bin.convert(aug.mode)  # restore original colour mode

    return aug_out, angle, tx, ty
