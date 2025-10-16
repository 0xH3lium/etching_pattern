from typing import Union
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders the four corner points of a rectangle.

    Args:
        pts: Array of 4 corner points

    Returns:
        Ordered corner points
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_surface_data(image_path: str) -> np.ndarray:
    """Extracts a 2D data matrix from an image of a 3D surface plot.

    Args:
        image_path: Path to the input image

    Returns:
        2D data matrix with normalized values
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find any data contours in the image.")
    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    src_pts = order_points(box)
    (tl, tr, br, bl) = src_pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    dst_pts = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(image, M, (max_width, max_height))
    warped_mask = cv2.warpPerspective(mask, M, (max_width, max_height))
    warped_hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    hue_channel = warped_hsv[:, :, 0]
    valid_hues = hue_channel[warped_mask > 0]
    if len(valid_hues) == 0:
        raise ValueError("Warped image contains no valid data pixels.")
    h_min, h_max = np.min(valid_hues), np.max(valid_hues)
    result_matrix = np.full(hue_channel.shape, np.nan, dtype=np.float32)
    if h_max - h_min > 0:
        normalized_values = (h_max - hue_channel[warped_mask > 0]) / (h_max - h_min)
    else:
        normalized_values = 0.5
    result_matrix[warped_mask > 0] = normalized_values
    return result_matrix