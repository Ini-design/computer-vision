import cv2
import numpy as np

def preprocess_motion_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert a video frame to grayscale and reduce noise.
    """
    if frame is None:
        raise ValueError("Frame is empty.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return blurred

def calculate_motion_mask(previous_frame: np.ndarray, current_frame: np.ndarray, threshold: int = 25) -> np.ndarray:
    """
    Compare two frames and create a binary motion mask.
    """
    if previous_frame is None:
        raise ValueError("Previous frame is empty.")
    if current_frame is None:
        raise ValueError("Current frame is empty.")
    previous = preprocess_motion_frame(previous_frame)
    current = preprocess_motion_frame(current_frame)
    difference = cv2.absdiff(previous, current)
    _, motion_mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    # Fill small gaps in detected motion.
    kernel = np.ones((5, 5), dtype=np.uint8 )
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    return motion_mask

def detect_motion_regions(
motion_mask: np.ndarray, min_area: int = 500) -> list:
    """
    Find bounding boxes around regions with detected motion.
    """
    if motion_mask is None:
        raise ValueError("Motion mask is empty.")
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area": area,
        })
    return regions

def draw_motion_boxes(frame: np.ndarray, regions: list) -> np.ndarray:
    """
    Draw bounding boxes around detected moving regions.
    """
    if frame is None:
        raise ValueError("Frame is empty.")
    result = frame.copy()
    for region in regions:
        x = region["x"]
        y = region["y"]
        width = region["width"]
        height = region["height"]
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(result, "Motion", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return result

def process_motion(previous_frame: np.ndarray, current_frame: np.ndarray, threshold: int = 25, min_area: int = 500) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Process two consecutive frames for motion detection.
    Returns:
        motion_mask
        processed_frame
        motion_regions
        processed_frame_with_boxes
    """
    motion_mask = calculate_motion_mask(previous_frame, current_frame, threshold)
    regions = detect_motion_regions(motion_mask, min_area)
    result = draw_motion_boxes(current_frame, regions)
    return (motion_mask, current_frame, regions, result)


#object tracking
def create_tracker(tracker_type: str = "CSRT"):
    """
    Create an OpenCV object tracker.
    """
    tracker_type = tracker_type.upper()
    if tracker_type == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerCSRT_create()
    elif tracker_type == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerKCF_create()
    raise ValueError(f"Unsupported tracker: {tracker_type}")
def initialize_tracker(frame: np.ndarray, bbox: tuple, tracker_type: str = "CSRT"):
    """
    Initialize an object tracker with a selected ROI.
    """
    if frame is None:
        raise ValueError("Frame is empty.")
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain x, y, width, height.")
    x, y, width, height = map(int, bbox)
    frame_height, frame_width = frame.shape[:2]
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(1, min(width, frame_width - x))
    height = max(1, min(height, frame_height - y))
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box dimensions must be positive.")
    
    tracker = create_tracker(tracker_type)
    bbox = (x, y, width, height)
    try:
        tracker.init(frame, bbox)
    except cv2.error as e:
        raise RuntimeError(f"Failed to initialize tracker: {e}")
    return tracker
def update_tracker(tracker, frame: np.ndarray) -> tuple[bool, tuple | None]:
    """
    Update the tracker using a new frame.
    Returns:
        success
        bounding box
    """
    if frame is None:
        raise ValueError("Frame is empty.")
    success, bbox = tracker.update(frame)
    if not success:
        return False, None
    return True, tuple(map(int, bbox))
def draw_tracking_box(frame: np.ndarray, bbox: tuple | None, success: bool) -> np.ndarray:
    """
    Draw the current tracking bounding box.
    """
    if frame is None:
        raise ValueError("Frame is empty.")
    result = frame.copy()
    if not success or bbox is None:
        cv2.putText(result, "Tracking lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return result
    x, y, width, height = bbox
    cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(result, "Tracking", (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result