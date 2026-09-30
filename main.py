"""Application coordinator: capture -> detect -> update state -> scroll -> draw."""

# Report missing top-level packages clearly; preserve unrelated import errors.
try:
    import cv2
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    raise SystemExit(
        "OpenCV is required. Install it in this Python environment with: "
        "python -m pip install opencv-python"
    ) from exc
try:
    import mediapipe as mp
except ModuleNotFoundError as exc:
    if exc.name != "mediapipe":
        raise
    raise SystemExit(
        "MediaPipe is required. Install it in this Python environment with: "
        "python -m pip install mediapipe"
    ) from exc
import time
import camera
import gesture_control as gestures
import gui


# Short names for MediaPipe's Tasks API classes.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# The model path is relative to the working directory; run from this folder.
# Detect two hands so the loop can recognize ambiguity and pause control.
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# Model loading happens here; a missing/incompatible .task file fails at startup.
landmarker = HandLandmarker.create_from_options(options)

def main():
    """Run camera processing until Q, a held fist, capture failure, or fail-safe."""
    # Keep one state object outside the loop so timers survive across frames.
    cap = camera.setup_camera()
    timestamp_ms = 0
    scroll_state = gestures.ScrollState()

    try:
        while cap.isOpened() and not gui.exit_requested():
            # Read one frame; bounded retries allow a temporary capture failure.
            success, frame = cap.read()
            attempts = 0

            while not success and attempts <= 5:  # caps our attempts to retry getting frames
                time.sleep(1)
                attempts += 1
                success, frame = cap.read()

            if not success:
                print("Frame is not being captured in main")
                break

            # Prepare the frame
            image, mp_image = camera.prepare_frame(frame) # image -> mirroring, mp_image -> mediapipe

            # Strictly increasing milliseconds for MediaPipe; gesture timers use seconds.
            timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))

            # detect landmarks
            result = camera.track_hands(
                landmarker,
                mp_image,
                timestamp_ms
            )

            # Stop before sending any more scroll commands after window closure.
            if gui.exit_requested():
                break

            # Exactly one visible hand controls scrolling; ambiguity pauses tracking.
            hand = result.hand_landmarks[0] if len(result.hand_landmarks) == 1 else None
            now = time.monotonic()
            # Update recognition before checking exit, which uses current_gesture.
            previous_y = scroll_state.previous_y
            ready = gestures.update_scroll_mode(scroll_state, hand, now)
            if gestures.update_exit_hold(scroll_state, now):
                break  # Normal cleanup below releases the camera and closes the GUI.
            if ready:
                # Only landmark 8 supplies movement; other finger positions only
                # contribute to recognizing activation and fist gestures.
                gestures.detect_vertical_movement(
                    scroll_state, hand[gestures.INDEX_FINGER_TIP].y, now)
            else:
                gestures.debug_scroll_state(
                    scroll_state,
                    hand[gestures.INDEX_FINGER_TIP].y if hand is not None else None,
                    previous_y,
                    f"mode gate: {len(result.hand_landmarks)} hands, "
                    f"activation={scroll_state.activation_elapsed:.2f}s")

            # Tick independently of movement, but pause for fists or missing hands.
            direction = gestures.continuous_scroll_direction(scroll_state, now, ready)
            if direction:
                try:
                    gestures.scroll(direction, scroll_state.scroll_amount)
                except gestures.pyautogui.FailSafeException:
                    print("Mouse-corner fail-safe triggered. Closing gesture control.")
                    break

            # Translate internal state into display labels; gui.py only draws them.
            gesture = scroll_state.current_gesture
            if len(result.hand_landmarks) > 1:
                gesture = "Multiple Hands"
            action = "Scroll Disabled"
            display_direction = 0
            if scroll_state.scroll_mode_active:
                action = "Move index finger up or down" if ready else "Paused"
                if ready and scroll_state.current_scroll_direction is not None:
                    action = f"Continuous Scrolling {scroll_state.current_scroll_direction.title()}"
                    display_direction = 1 if scroll_state.current_scroll_direction == "UP" else -1

            # Countdown/deactivation messages take priority over ordinary status.
            if scroll_state.activation_started_at is not None:
                action = (f"Activating... {scroll_state.activation_elapsed:.1f}s / "
                          f"{scroll_state.activation_hold_seconds:.1f}s")
            if scroll_state.current_gesture == "Fist" and scroll_state.scroll_mode_active:
                action = "Deactivating"
                display_direction = 0
            elif scroll_state.exit_seconds_remaining is not None:
                action = f"Exiting in {scroll_state.exit_seconds_remaining:.1f}s"
                display_direction = 0

            # Supply display state to the GUI; all drawing stays in gui.py.
            for detected_hand in result.hand_landmarks:
                gui.draw_hand_status(image, detected_hand)
            direction_label = scroll_state.current_scroll_direction
            reverse_label = None
            if scroll_state.reverse_frame_count:
                opposite = "UP" if direction_label == "DOWN" else "DOWN"
                reverse_label = f"{opposite} {scroll_state.reverse_frame_count}/{gestures.DIRECTION_REVERSE_FRAMES}"
            gui.draw_status_panel(
                image, scroll_state.scroll_mode_active, gesture, action, direction_label,
                reverse_label)
            gui.draw_scroll_direction(image, display_direction)
            if gui.show_frame(image):
                break
    finally:
        # Release resources on window close, normal exit, or an exception.
        cap.release()
        gui.close()
        landmarker.close()



if __name__ == "__main__":
    # Start the loop when launched as a script.
    main()