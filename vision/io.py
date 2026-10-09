import cv2
import numpy as np
from pathlib import Path


def load_image(file_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes into an OpenCV image.
    Returns:
        np.ndarray: Image in BGR format.
    """
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the image.")
    return image

def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file using OpenCV.
    Returns:
        cv2.VideoCapture: Open video capture object.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to open the video.")
    return capture

def get_image_info(image: np.ndarray) -> dict:
    """
    Extract basic information about an image.
    """
    if image is None:
        raise ValueError("Image is empty.")
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": image.shape,
        "dtype": str(image.dtype),
    }

def save_image(image: np.ndarray, output_path: str) -> str:
    """
    Save an OpenCV image to disk.
    """
    if image is None:
        raise ValueError("Cannot save an empty image.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save image to {output_path}")
    return str(path)

def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    """
    Encode an OpenCV image into bytes.
    Useful for Streamlit download buttons.
    """
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Unable to encode image.")
    return buffer.tobytes()

def capture_webcam_frame(camera_index: int = 0):
    """
    Capture one frame from a connected webcam.
    Returns:
        np.ndarray: Captured frame.
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise ValueError("Unable to access the webcam.")
    success, frame = camera.read()
    camera.release()
    if not success:
        raise ValueError("Unable to capture a webcam frame.")
    return frame
def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise ValueError("Unable to access the webcam.")

    return camera