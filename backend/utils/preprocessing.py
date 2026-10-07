"""
Preprocessing utilities.
Image transforms and face crop helpers.
"""

import numpy as np
from PIL import Image
import cv2


def crop_face(image: np.ndarray, bbox: list, margin: float = 0.2) -> np.ndarray:
    """
    Crop a face from the image with margin, ensuring square crop.

    Args:
        image: HxWx3 numpy array
        bbox: [x1, y1, x2, y2] coordinates
        margin: Fractional margin around face

    Returns:
        Cropped face as HxWx3 numpy array
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(c) for c in bbox]

    face_w = x2 - x1
    face_h = y2 - y1
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)

    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)

    # Make square
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return crop
