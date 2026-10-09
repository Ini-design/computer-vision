import cv2
import numpy as np

def canny_edges(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200,) -> np.ndarray:
    """
    Detect edges using the Canny edge detector.
    """
    if image is None:
        raise ValueError("Image is empty.")
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds cannot be negative.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def sobel_edges(image: np.ndarray, kernel_size: int = 3,) -> np.ndarray:
    """
    Detect edges using the Sobel operator.
    """
    if image is None:
        raise ValueError("Image is empty.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # magnitude = cv2.magnitude(cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y))
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    return magnitude
def find_contours(image: np.ndarray, threshold: int = 127,) -> list:
    """
    Detect contours from an image.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours( binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def measure_contour(contour) -> dict:
    """
    Calculate measurements for a single contour.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(
            moments["m01"] / moments["m00"])
    else:
        centroid_x = 0
        centroid_y = 0

    return {
        "area": area,
        "perimeter": perimeter,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "centroid": (centroid_x, centroid_y),
}
def draw_bounding_boxes(image: np.ndarray, contours: list, min_area: float = 100.0) -> np.ndarray:
    """
    Draw bounding boxes around detected objects.
    """
    if image is None:
        raise ValueError("Image is empty.")
    result = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return result
def count_objects(contours: list, min_area: float = 100.0) -> int:
    """
    Count contours whose area is above the minimum threshold.
    """
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            count += 1
    return count
def prepare_document(image: np.ndarray) -> np.ndarray:
    """
    Prepare an image for document boundary detection.
    """
    if image is None:
        raise ValueError("Image is empty.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    return edges
def find_document_contour(edges: np.ndarray) -> np.ndarray | None:
    """
    Find the largest four-sided contour,
    assumed to be the document boundary.
    """
    if edges is None:
        raise ValueError("Edges image is empty.")
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4 and cv2.contourArea(approximation) > 250000:
            return approximation
    return None
def order_document_points(points: np.ndarray) -> np.ndarray:
    """
    Order four document corners as:
    top-left, top-right, bottom-right, bottom-left.
    """
    points = points.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered
def perspective_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform the document into a top-down view.
    """
    if image is None:
        raise ValueError("Image is empty.")
    rect = order_document_points(points)
    top_left, top_right, bottom_right, bottom_left = rect
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))
    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    max_height = int(max(height_right, height_left))
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform( rect, destination)
    scanned = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return scanned
def scan_document(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Detect and perspective-correct a document.
    Returns:
        edges,
        document_contour,
        scanned_document
    """
    edges = prepare_document(image)
    document_contour = find_document_contour(edges)
    if document_contour is None:
        raise ValueError("No four-sided document boundary was detected.")
    scanned = perspective_transform(image, document_contour)
    return edges, document_contour, scanned
