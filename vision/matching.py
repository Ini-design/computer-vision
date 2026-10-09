import cv2
import numpy as np

def detect_orb_features(image: np.ndarray, n_features: int = 1000):
    """
    Detect ORB keypoints and compute descriptors.
    """
    if image is None:
        raise ValueError("Image is empty.")
    if n_features < 100:
        raise ValueError("n_features must be at least 100.")
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, max_matches: int = 50):
    """
    Match ORB descriptors using
    Brute-Force Hamming distance.
    """
    if descriptors1 is None:
        return []
    if descriptors2 is None:
        return []
    if len(descriptors1) == 0:
        return []
    if len(descriptors2) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True )
    matches = matcher.match(descriptors1, descriptors2)
    # Smaller distance = better match
    matches = sorted(matches, key=lambda match: match.distance)
    return matches[:max_matches]
def draw_keypoints(image: np.ndarray, keypoints):
    """
    Draw detected ORB keypoints.
    """
    if image is None:
        raise ValueError("Image is empty.")
    return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS )

def draw_matches(image1: np.ndarray,
    keypoints1, image2: np.ndarray, keypoints2, matches):
    """
    Draw feature matches between two images.
    """
    if image1 is None or image2 is None:
        raise ValueError( "One or both images are empty.")
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

def calculate_match_quality(matches):
    """
    Calculate simple matching statistics.
    """
    if not matches:
        return {
            "match_count": 0,
            "average_distance": None,
            "good_matches": 0,
        }
    distances = [ match.distance for match in matches]
    average_distance = float(np.mean(distances))
    # Lower Hamming distance indicates
    # stronger descriptor similarity.
    good_matches = sum(1 for distance in distances if distance < 50 )
    return {
        "match_count": len(matches),
        "average_distance": average_distance,
        "good_matches": good_matches,
    }

def feature_match(image1: np.ndarray, image2: np.ndarray, n_features: int = 1000, max_matches: int = 50):
    """
    Complete ORB feature matching pipeline.
    """
    keypoints1, descriptors1 = (detect_orb_features( image1, n_features))
    keypoints2, descriptors2 = ( detect_orb_features(image2, n_features ))
    matches = match_features( descriptors1, descriptors2, max_matches)
    visualization = draw_matches( image1, keypoints1, image2, keypoints2, matches)
    quality = calculate_match_quality( matches)
    return {
        "keypoints1": keypoints1,
        "keypoints2": keypoints2,
        "descriptors1": descriptors1,
        "descriptors2": descriptors2,
        "matches": matches,
        "visualization": visualization,
        "quality": quality,
    }