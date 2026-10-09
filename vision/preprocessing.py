import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.
    """
    if image is None:
        raise ValueError("Image is empty.")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
    """
    Adjust image brightness.
    value:
        Negative -> darker
        Positive -> brighter
    """
    if image is None:
        raise ValueError("Image is empty.")
    if not -255 <= value <= 255:
        raise ValueError("Brightness value must be between -255 and 255.")
    result = cv2.convertScaleAbs(image, alpha=1.0, beta=value)
    return result
def adjust_contrast(image: np.ndarray, value: float) -> np.ndarray:
    """
    Adjust image contrast.
    value:
        1.0 = original contrast
        < 1.0 = lower contrast
        > 1.0 = higher contrast
    """
    if image is None:
        raise ValueError("Image is empty.")
    if value < 0:
        raise ValueError("Contrast value cannot be negative.")
    return cv2.convertScaleAbs(image, alpha=value, beta=0)

def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.
    """
    if image is None:
        raise ValueError("Image is empty.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median blur for noise reduction.
    """
    if image is None:
        raise ValueError("Image is empty.")

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")
    return cv2.medianBlur(image, kernel_size)

def bilateral_filter(image: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filtering.
    Bilateral filtering reduces noise while preserving edges.
    """
    if image is None:
        raise ValueError("Image is empty.")
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def calculate_histogram(image: np.ndarray) -> np.ndarray:
    """
    Calculate a grayscale histogram.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = to_grayscale(image)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])  
    return histogram

def histogram_equalization(
    image: np.ndarray
) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = to_grayscale(image)
    return cv2.equalizeHist(gray)

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE histogram enhancement.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray)

def global_threshold(image: np.ndarray, threshold: int = 127, max_value: int = 255) -> np.ndarray:
    """
    Apply global binary thresholding.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = to_grayscale(image)
    _, result = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
    return result
def otsu_threshold(image: np.ndarray, max_value: int = 255) -> np.ndarray:
    """
    Apply Otsu thresholding.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = to_grayscale(image)
    _, result = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result
def adaptive_threshold(image: np.ndarray, block_size: int = 11, constant: int = 2) -> np.ndarray:
    """
    Apply adaptive Gaussian thresholding.
    """
    if image is None:
        raise ValueError("Image is empty.")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("Block size must be an odd number >= 3.")
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)