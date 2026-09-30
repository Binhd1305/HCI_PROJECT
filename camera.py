"""Camera capture and image conversion; gesture decisions live in gesture_control."""

import cv2
import mediapipe as mp


def setup_camera():
    """Open webcam 0 and request 640x480; the camera may choose another size."""
    # If no frames arrive, check the camera index and whether another app uses it.
    cam = cv2.VideoCapture(0) # 0 for default webcam
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam

def prepare_frame(frame):
    """Return a mirrored BGR display frame and an RGB MediaPipe input image."""
    image = cv2.flip(frame, 1)  # flip the captured frame for mirroring

    # OpenCV stores BGR, but MediaPipe expects RGB; wrong colors hurt detection.
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OpenCV image -> MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    return image, mp_image

def track_hands(landmarker, mp_image, timestamp_ms):
    """
    Run detection on one frame and return MediaPipe's result object.
    result.hand_landmarks is empty when no hand is detected.
    VIDEO mode requires a strictly increasing timestamp in milliseconds.
    """

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    return result


