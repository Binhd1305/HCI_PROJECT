"""Simple hand gestures and state for vertical scrolling."""

import math
from dataclasses import dataclass
import pyautogui

# A mouse pointer in a screen corner makes PyAutoGUI raise FailSafeException.
# main.py catches that exception and leaves the loop through normal cleanup.
pyautogui.FAILSAFE = True

# Amount controls distance per tick; interval controls continuous scrolling rate.
SCROLL_AMOUNT = 10  # Base wheel clicks for small movements outside the dead zone.
SCROLL_INTERVAL = 0.06  # Seconds between continuous scroll commands.
MOVEMENT_THRESHOLD = 0.02  # Fraction of the camera frame height.
DIRECTION_CHANGE_THRESHOLD = 0.06
DIRECTION_REVERSE_FRAMES = 4
HAND_LOSS_RESET_FRAMES = 5
DEBUG_SCROLL = True  # Temporary console diagnostics; set False after debugging.

INDEX_FINGER_TIP = 8  # MediaPipe index; wrist is 0 and is not our scroll input.


def _distance(a, b):
    """Return Euclidean distance between two landmarks using x, y, and z."""
    return math.sqrt(sum((getattr(a, axis) - getattr(b, axis)) ** 2
                         for axis in ("x", "y", "z")))


def _angle(a, joint, b):
    """Return the joint angle in degrees; a straight finger approaches 180."""
    # Build vectors pointing from the joint toward the two surrounding points.
    first = [getattr(a, axis) - getattr(joint, axis) for axis in ("x", "y", "z")]
    second = [getattr(b, axis) - getattr(joint, axis) for axis in ("x", "y", "z")]
    length = _distance(a, joint) * _distance(b, joint)
    # Coincident points cannot define an angle; avoid dividing by zero.
    if length < 1e-8:
        return 0
    cosine = sum(x * y for x, y in zip(first, second)) / length
    # Clamp rounding errors so acos never receives a value outside [-1, 1].
    return math.degrees(math.acos(max(-1, min(1, cosine))))


def count_fingers(hand_landmarks):
    """Count straight fingers using joint angles, for either hand."""
    hand = hand_landmarks
    count = 0
    # Index, middle, ring, pinky: require a straight joint and a tip farther
    # from the wrist than the middle joint. These are heuristic pose thresholds.
    for base, middle, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        if (_angle(hand[base], hand[middle], hand[tip]) > 155
                and _distance(hand[tip], hand[0]) > _distance(hand[middle], hand[0])):
            count += 1

    # A straight thumb must also spread away from the index knuckle.
    palm_width = _distance(hand[5], hand[17])
    if (_angle(hand[1], hand[2], hand[3]) > 150
            and _angle(hand[2], hand[3], hand[4]) > 150
            and _distance(hand[4], hand[5]) > 0.6 * palm_width):
        count += 1
    return count


def is_fist(hand_landmarks):
    """Treat zero extended fingers as a fist (the same rule used by mode updates)."""
    return count_fingers(hand_landmarks) == 0


@dataclass
class ScrollState:
    """Memory shared across frames; create one instance per application run."""
    # Mode stays on until a confirmed fist. None means no movement baseline yet.
    scroll_mode_active: bool = False
    previous_y: float | None = None
    # Consecutive-frame confirmation is used for fist deactivation.
    pending_gesture: str | None = None
    gesture_frames: int = 0
    stable_frames: int = 3
    # Y is normalized image position, while all times below are seconds.
    movement_threshold: float = MOVEMENT_THRESHOLD
    direction_change_threshold: float = DIRECTION_CHANGE_THRESHOLD
    current_scroll_direction: str | None = None
    reverse_frame_count: int = 0
    reverse_start_y: float | None = None
    hand_lost_frames: int = 0
    scroll_interval: float = SCROLL_INTERVAL
    last_scroll_time: float = -math.inf
    scroll_amount: int = SCROLL_AMOUNT
    # Keep a scroll message visible briefly, even between wheel commands.
    action_text: str = ""
    action_until: float = 0
    current_gesture: str = "No Gesture"
    # Exit and activation have separate timers; stopping scrolling must not
    # erase the continuous fist hold used to quit the application.
    fist_started_at: float | None = None
    exit_hold_seconds: float = 3.0
    exit_seconds_remaining: float | None = None
    activation_started_at: float | None = None
    activation_hold_seconds: float = 2.0
    activation_elapsed: float = 0.0


def update_exit_hold(state, now):
    """Call after gesture detection; require a continuously visible fist."""
    # Any other pose or missing hand cancels the quit countdown.
    if state.current_gesture != "Fist":
        state.fist_started_at = None
        state.exit_seconds_remaining = None
        return False
    # Save the first frame's time, not the current time on every frame.
    if state.fist_started_at is None:
        state.fist_started_at = now
    state.exit_seconds_remaining = max(
        0.0, state.exit_hold_seconds - (now - state.fist_started_at)
    )
    return state.exit_seconds_remaining == 0.0


def reset_tracking(state, reset_direction=True):
    """Clear motion and transient feedback without changing mode or hold timers."""
    # Reacquiring a hand should establish a baseline instead of causing a jump.
    state.previous_y = None
    state.last_scroll_time = -math.inf
    state.scroll_amount = SCROLL_AMOUNT
    state.action_text = ""
    state.action_until = 0
    state.reverse_frame_count = 0
    state.reverse_start_y = None
    if reset_direction:
        state.current_scroll_direction = None


def update_scroll_mode(state, hand_landmarks, now):
    """Hold five fingers for two seconds to activate; confirm fists by frames."""
    # now comes from time.monotonic(), so system clock changes do not affect holds.
    # None also represents multiple visible hands, as decided by main.py.
    fingers = count_fingers(hand_landmarks) if hand_landmarks is not None else None
    if hand_landmarks is None:
        state.hand_lost_frames += 1
        if state.hand_lost_frames >= HAND_LOSS_RESET_FRAMES:
            state.current_scroll_direction = None
            state.reverse_frame_count = 0
            state.reverse_start_y = None
    else:
        state.hand_lost_frames = 0
    state.current_gesture = ("No Gesture" if fingers is None else
                             "Open Hand" if fingers == 5 else
                             "Fist" if fingers == 0 else f"{fingers} Fingers")
    gesture = "open" if fingers == 5 else "fist" if fingers == 0 else None
    # Break consecutive-frame confirmation whenever the recognized pose changes.
    if gesture is None:
        state.pending_gesture = None
        state.gesture_frames = 0
    elif gesture == state.pending_gesture:
        state.gesture_frames += 1
    else:
        state.pending_gesture = gesture
        state.gesture_frames = 1

    # Only activation needs five fingers. Other poses keep the mode latched.
    if not state.scroll_mode_active and gesture == "open":
        if state.activation_started_at is None:
            state.activation_started_at = now
        state.activation_elapsed = now - state.activation_started_at
        if state.activation_elapsed >= state.activation_hold_seconds:
            # Activation starts fresh: the next movement sample becomes baseline.
            state.scroll_mode_active = True
            state.activation_started_at = None
            state.activation_elapsed = 0.0
            reset_tracking(state)
    else:
        # Missing hands, multiple hands, or any other pose interrupt the hold.
        state.activation_started_at = None
        state.activation_elapsed = 0.0

    # Confirmed fist switches the mode off; activation timing was cleared above.
    if gesture == "fist" and state.gesture_frames >= state.stable_frames:
        state.scroll_mode_active = False
        reset_tracking(state)

    # Pause immediately for a possible fist or missing hand, even while
    # waiting for the stable fist confirmation to switch the mode off.
    if hand_landmarks is None or gesture == "fist":
        reset_tracking(state, reset_direction=gesture == "fist")
        return False
    # True means main.py may process index motion on this frame, not just mode ON.
    return state.scroll_mode_active


def debug_scroll_state(state, current_y, previous_y, reason, direction=0):
    """Report the input baseline, including frames that never reach wheel output."""
    if not DEBUG_SCROLL:
        return
    delta = None if current_y is None or previous_y is None else previous_y - current_y
    def fmt(value):
        return "None" if value is None else f"{value:.4f}"
    print(
        f"Mode={state.scroll_mode_active}, Dir={state.current_scroll_direction}, "
        f"Gesture={state.current_gesture}, current_y={fmt(current_y)}, "
        f"previous_y={fmt(previous_y)}, delta_y={fmt(delta)}, "
        f"movement={fmt(None if delta is None else abs(delta))}, "
        f"reverse_frame_count={state.reverse_frame_count}, "
        f"scroll_amount={state.scroll_amount}, Reason={reason}, "
        f"MotionResult={direction}, Wheel=handled by continuous timer"
    )


def detect_vertical_movement(state, current_y, now):
    """Update the direction lock only; movement never schedules wheel commands."""
    previous_y = state.previous_y

    def finish(direction, reason):
        debug_scroll_state(state, current_y, previous_y, reason, direction)
        return direction

    if not state.scroll_mode_active:
        state.previous_y = None
        state.current_scroll_direction = None
        state.reverse_frame_count = 0
        state.reverse_start_y = None
        return finish(0, "mode disabled")
    # First sample after activation or tracking loss cannot define movement.
    if state.previous_y is None:
        state.previous_y = current_y
        return finish(0, "establishing baseline")
    difference = state.previous_y - current_y  # Image Y increases downward.
    movement = abs(difference)
    if movement == 0:
        state.reverse_frame_count = 0
        state.reverse_start_y = None
        state.previous_y = current_y
        return finish(0, "no movement")
    direction = 1 if difference > 0 else -1
    direction_name = "UP" if direction > 0 else "DOWN"

    if state.current_scroll_direction is None:
        if movement <= state.movement_threshold:
            return finish(0, "dead zone before direction lock")
        state.current_scroll_direction = direction_name
    elif direction_name != state.current_scroll_direction:
        if state.reverse_start_y is None:
            state.reverse_start_y = state.previous_y
        reverse_movement = (
            state.reverse_start_y - current_y
            if state.current_scroll_direction == "DOWN"
            else current_y - state.reverse_start_y
        )
        state.previous_y = current_y
        if reverse_movement <= state.direction_change_threshold:
            state.reverse_frame_count = 0
            return finish(0, f"reversal distance {reverse_movement:.4f} below threshold")
        state.reverse_frame_count += 1
        if state.reverse_frame_count < DIRECTION_REVERSE_FRAMES:
            return finish(0, "waiting for four reversal frames")
        state.current_scroll_direction = direction_name
        state.reverse_frame_count = 0
        state.reverse_start_y = None
        # A confirmed reversal takes effect immediately, even between timer ticks.
        state.last_scroll_time = -math.inf
    else:
        state.reverse_frame_count = 0
        state.reverse_start_y = None
        if movement <= state.movement_threshold:
            # Keep the baseline so small movements in the locked direction accumulate.
            return finish(0, "dead zone in locked direction")

    # Consume movement without touching the independent wheel timer.
    state.previous_y = current_y
    return finish(direction, "movement accepted")


def continuous_scroll_direction(state, now, ready=True):
    """Schedule one wheel tick from the lock, including stationary frames.

    The caller supplies monotonic seconds and tracking readiness every frame.
    Never send a catch-up burst after a slow frame or a tracking interruption.
    """
    if not ready or not state.scroll_mode_active or state.current_scroll_direction is None:
        return 0
    if now - state.last_scroll_time < state.scroll_interval:
        return 0
    state.last_scroll_time = now
    direction = 1 if state.current_scroll_direction == "UP" else -1
    state.action_text = f"Continuous Scrolling {state.current_scroll_direction.title()}"
    state.action_until = now + 0.5
    return direction


def scroll(direction, amount=None):
    """Scroll the browser: keep it focused and the pointer over the webpage."""
    # Positive wheel clicks scroll up; negative clicks scroll down. No browser
    # automation is involved, so check focus/pointer position if nothing moves.
    if direction:
        amount = SCROLL_AMOUNT if amount is None else amount
        # Our cooldown handles pacing; skip PyAutoGUI's extra post-call sleep.
        # The corner fail-safe still runs before the wheel command.
        if DEBUG_SCROLL:
            try:
                print(f"ActiveWindow={pyautogui.getActiveWindowTitle()!r}, "
                      f"Pointer={pyautogui.position()}")
            except Exception as exc:
                # Diagnostic failures must never suppress a wheel command.
                print(f"Window/pointer diagnostics unavailable: {exc}")
            print("SCROLL UP" if direction > 0 else "SCROLL DOWN", amount)
        pyautogui.scroll(amount if direction > 0 else -amount, _pause=False)
        if DEBUG_SCROLL:
            print("pyautogui.scroll() called and returned")
