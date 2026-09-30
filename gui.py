"""Camera display only: receives state without recognizing gestures."""

import cv2

# OpenCV colors are BGR, not RGB. Drawing functions modify the supplied frame.
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (240, 240, 240)
MUTED = (190, 195, 205)
GREEN = (100, 220, 100)
RED = (90, 90, 240)
YELLOW = (80, 220, 255)
WINDOW_NAME = "Hand Gesture Control"
_window_created = False
_exit_requested = False


def exit_requested():
    """Remember a closed window so a later imshow cannot recreate it."""
    global _exit_requested
    if _window_created and not _exit_requested:
        try:
            _exit_requested = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            # Some OpenCV backends raise when the user destroys the window.
            _exit_requested = True
    return _exit_requested


def _text(frame, text, position, color=WHITE, scale=0.5):
    """Draw antialiased text; position is the text's bottom-left baseline."""
    cv2.putText(frame, text, position, FONT, scale, color, 1, cv2.LINE_AA)


def draw_status_panel(frame, scroll_mode_active, gesture, action, direction=None, reverse=None):
    """Draw a translucent panel; coordinates target the 640x480 camera feed."""
    # Blend a dark rectangle with the live feed so the camera remains visible.
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (365, 325), (25, 28, 34), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)
    # Mode color appears in both the border and status dot.
    color = GREEN if scroll_mode_active else RED
    cv2.rectangle(frame, (10, 10), (14, 325), color, -1)
    _text(frame, "HAND GESTURE CONTROL", (26, 38), scale=0.65)
    cv2.line(frame, (26, 51), (348, 51), MUTED, 1)
    cv2.circle(frame, (32, 76), 6, color, -1, cv2.LINE_AA)
    # Render labels supplied by main.py without making gesture decisions.
    mode = "ON" if scroll_mode_active else "OFF"
    _text(frame, f"SCROLL MODE: {mode}", (48, 82), color, 0.58)
    _text(frame, f"Gesture: {gesture}", (26, 113))
    _text(frame, f"Direction: {direction or 'NONE'}", (26, 144), YELLOW, 0.47)
    _text(frame, f"Action: {action}", (26, 169), YELLOW, 0.42)
    control = (f"Reverse: {reverse}" if reverse else
               "Control: Index Finger" if scroll_mode_active else "Hold 5 fingers to activate")
    _text(frame, control, (26, 194), MUTED, 0.47)
    cv2.line(frame, (26, 211), (348, 211), MUTED, 1)
    draw_instructions(frame)
    return frame


def draw_instructions(frame):
    """Draw the quick-start guide below the current status."""
    # If hold durations change in gesture_control.py, update these labels too.
    _text(frame, "Hold 5 fingers for 2s to activate", (26, 235), MUTED, 0.47)
    _text(frame, "Move index UP/DOWN; hold to continue", (26, 258), MUTED, 0.44)
    _text(frame, "Make a fist to deactivate", (26, 281), MUTED, 0.47)
    _text(frame, "Hold fist 3s = Quit | Q = Quit", (26, 308), MUTED, 0.43)
    return frame


def draw_hand_status(frame, hand_landmarks):
    """Draw the supplied landmarks without interpreting their pose."""
    height, width = frame.shape[:2]
    # Convert normalized MediaPipe coordinates to camera-frame pixels.
    points = [(int(point.x * width), int(point.y * height))
              for point in hand_landmarks]
    # Connect each finger and the palm using MediaPipe's landmark indices.
    for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
                  (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (17, 0)):
        for start, end in zip(chain, chain[1:]):
            cv2.line(frame, points[start], points[end], GREEN, 2, cv2.LINE_AA)
    # Draw dots last so joint positions remain visible above the connections.
    for point in points:
        cv2.circle(frame, point, 3, WHITE, -1, cv2.LINE_AA)
    return frame


def draw_scroll_direction(frame, direction):
    """Draw up for +1, down for -1, and nothing for zero."""
    # Draw an actual arrow because OpenCV's default font lacks Unicode arrows.
    if direction:
        x = 337
        start_y, end_y = (149, 124) if direction > 0 else (124, 149)
        cv2.arrowedLine(frame, (x, start_y), (x, end_y), YELLOW,
                        2, cv2.LINE_AA, tipLength=0.4)
    return frame


def show_frame(frame):
    """Display the composed frame and return whether the user requested exit."""
    global _window_created, _exit_requested
    # Check BEFORE imshow: otherwise clicking X can recreate the window.
    if exit_requested():
        return True
    draw_welcome(frame)
    cv2.imshow(WINDOW_NAME, frame)
    _window_created = True
    # waitKey services window events; Q is read when the camera window has focus.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        _exit_requested = True
    return exit_requested()


def draw_welcome(frame):
    """Keep a small welcome label aligned to the top-right corner."""
    text = "Welcome to HCI Team App"
    scale = 0.45
    # Measure the string so its right edge stays aligned with the frame margin.
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x = frame.shape[1] - width - 18
    y = 20 + height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, 10),
                  (frame.shape[1] - 10, y + baseline + 8), (25, 28, 34), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)
    _text(frame, text, (x, y), WHITE, scale)
    return frame


def close():
    """Close all OpenCV display windows during application cleanup."""
    global _exit_requested
    _exit_requested = True
    cv2.destroyAllWindows()
