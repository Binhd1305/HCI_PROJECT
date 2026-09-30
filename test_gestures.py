"""Regression checks with simulated poses/times and mocked mouse-wheel output.

Run with: .venv\\Scripts\\python.exe -m unittest test_gestures.py
These tests cover state transitions, not live camera recognition accuracy.
"""

import unittest
from unittest.mock import patch

import gesture_control as gestures


class ScrollTests(unittest.TestCase):
    """Check activation, movement, deactivation, and exit without a webcam."""
    def test_browser_scroll_commands_respect_mode_and_cooldown(self):
        """OS commands must respect mode, movement direction, and cooldown."""
        # Mock the OS output so tests never scroll the user's desktop.
        with patch("gesture_control.pyautogui.scroll") as wheel:
            def move(y, now):
                """Simulate the loop's movement-to-wheel-command path."""
                gestures.detect_vertical_movement(self.state, y, now)
                direction = gestures.continuous_scroll_direction(self.state, now)
                if direction:
                    gestures.scroll(direction, self.state.scroll_amount)

            # Disabled mode and tiny movements must produce no wheel commands.
            move(0.5, 0.0)
            move(0.2, 0.1)
            wheel.assert_not_called()
            self.activate()
            move(0.5, 2.0)
            move(0.49, 2.1)
            wheel.assert_not_called()
            # A larger upward move fires once; a move during cooldown is ignored.
            move(0.4, 2.2)
            wheel.assert_called_once_with(gestures.SCROLL_AMOUNT, _pause=False)
            move(0.6, 2.25)
            self.assertEqual(wheel.call_count, 1)
            # Reversal needs four consecutive strong opposite-motion frames.
            move(0.7, 2.4)
            move(0.8, 2.5)
            move(0.9, 2.6)
            self.assertEqual(wheel.call_args.args, (-gestures.SCROLL_AMOUNT,))
            self.show(0, frames=3, now=2.8)
            move(0.1, 3.0)
            self.assertEqual(wheel.call_count, 4)

    def test_fist_exits_only_after_three_seconds(self):
        """Check the exact three-second exit boundary with a simulated clock."""
        state = gestures.ScrollState(current_gesture="Fist")
        self.assertFalse(gestures.update_exit_hold(state, 10.0))
        self.assertFalse(gestures.update_exit_hold(state, 12.99))
        self.assertTrue(gestures.update_exit_hold(state, 13.0))

    def test_interrupted_fist_restarts_exit_countdown(self):
        """Changing pose or losing the hand must cancel accumulated exit time."""
        for interruption in ("Open Hand", "No Gesture", "2 Fingers"):
            state = gestures.ScrollState(current_gesture="Fist")
            gestures.update_exit_hold(state, 0.0)
            state.current_gesture = interruption
            self.assertFalse(gestures.update_exit_hold(state, 2.0))
            self.assertIsNone(state.exit_seconds_remaining)
            state.current_gesture = "Fist"
            self.assertFalse(gestures.update_exit_hold(state, 3.0))
            self.assertFalse(gestures.update_exit_hold(state, 5.9))
            self.assertTrue(gestures.update_exit_hold(state, 6.0))

    def setUp(self):
        """Give every test fresh state so tests cannot influence one another."""
        debug = patch.object(gestures, "DEBUG_SCROLL", False)
        debug.start()
        self.addCleanup(debug.stop)
        self.state = gestures.ScrollState()

    def show(self, fingers, frames=1, now=0.0):
        """Mock recognition for a number of frames at the supplied time."""
        with patch("gesture_control.count_fingers", return_value=fingers):
            for _ in range(frames):
                ready = gestures.update_scroll_mode(self.state, object(), now)
        return ready

    def activate(self):
        """Simulate open-hand observations two seconds apart to enable the mode."""
        self.show(5, now=0.0)
        self.show(5, now=2.0)

    def test_consecutive_frames_and_fist_stop(self):
        """Activation uses elapsed time; fist deactivation requires three frames."""
        self.show(5, 2)
        self.assertFalse(self.state.scroll_mode_active)
        self.show(2)
        self.show(5, 2)
        self.assertFalse(self.state.scroll_mode_active)
        self.show(5, now=2.0)
        self.assertTrue(self.state.scroll_mode_active)
        self.state.previous_y = 0.5
        self.assertFalse(self.show(0))
        self.assertIsNone(self.state.previous_y)
        self.assertTrue(self.state.scroll_mode_active)
        self.show(0, 2)
        self.assertFalse(self.state.scroll_mode_active)
        self.assertEqual(gestures.detect_vertical_movement(self.state, 0.1, 5), 0)

    def test_direction_dead_zone_and_reversal(self):
        """Direction recognition runs independently of wheel timing."""
        self.activate()
        move = lambda y, now: gestures.detect_vertical_movement(self.state, y, now)
        self.assertEqual(move(0.5, 0), 0)
        self.assertEqual(move(0.49, 0.1), 0)
        self.assertEqual(move(0.44, 0.2), 1)
        self.assertEqual(move(0.3, 0.24), 1)
        self.assertEqual(move(0.3, 0.5), 0)
        # One strong opposite frame cannot reverse the established UP lock.
        self.assertEqual(move(0.4, 0.6), 0)
        self.assertEqual(self.state.current_scroll_direction, "UP")
        self.assertEqual(move(0.5, 0.7), 0)
        self.assertEqual(move(0.6, 0.8), 0)
        self.assertEqual(move(0.7, 0.9), -1)
        self.assertEqual(self.state.current_scroll_direction, "DOWN")

    def test_weak_opposite_motion_is_ignored(self):
        """Small opposite jitter must not change the locked direction."""
        self.state.scroll_mode_active = True
        move = lambda y, now: gestures.detect_vertical_movement(self.state, y, now)
        self.assertEqual(move(0.5, 0.0), 0)
        self.assertEqual(move(0.6, 0.1), -1)
        self.assertEqual(move(0.55, 0.2), 0)
        self.assertEqual(self.state.current_scroll_direction, "DOWN")
        self.assertEqual(self.state.reverse_frame_count, 0)
        self.assertEqual(move(0.65, 0.3), -1)

    def test_gradual_reversal_accumulates_before_frame_confirmation(self):
        """Several small opposite steps cross the threshold, then need four frames."""
        self.state.scroll_mode_active = True
        move = lambda y, now: gestures.detect_vertical_movement(self.state, y, now)
        self.assertEqual(move(0.5, 0.0), 0)
        self.assertEqual(move(0.6, 0.1), -1)
        for index, y in enumerate((0.575, 0.55, 0.525, 0.50, 0.475), start=2):
            self.assertEqual(move(y, index / 10), 0)
            self.assertEqual(self.state.current_scroll_direction, "DOWN")
        self.assertEqual(move(0.45, 0.8), 1)
        self.assertEqual(self.state.current_scroll_direction, "UP")

    def test_missing_hand_and_fist_reset_direction(self):
        """Keep lock briefly through tracking loss, then clear it; fists clear it."""
        self.activate()
        self.state.current_scroll_direction = "DOWN"
        for frame in range(gestures.HAND_LOSS_RESET_FRAMES - 1):
            gestures.update_scroll_mode(self.state, None, frame)
            self.assertEqual(self.state.current_scroll_direction, "DOWN")
        gestures.update_scroll_mode(self.state, None, 5)
        self.assertIsNone(self.state.current_scroll_direction)

        self.state.current_scroll_direction = "UP"
        self.show(0)
        self.assertIsNone(self.state.current_scroll_direction)

    def test_initial_direction_and_fixed_amount_in_both_directions(self):
        """Movement selects direction; the configured amount sets continuous speed."""
        for movement, amount in ((0.019, None), (0.021, 10), (0.03, 10), (0.06, 10), (0.09, 10)):
            for direction in (-1, 1):
                self.state = gestures.ScrollState(scroll_mode_active=True)
                gestures.detect_vertical_movement(self.state, 0.5, 0.0)
                result = gestures.detect_vertical_movement(
                    self.state, 0.5 - direction * movement, 0.1)
                self.assertEqual(result, 0 if amount is None else direction)
                if amount is not None:
                    self.assertEqual(self.state.scroll_amount, amount)

    def test_interval_still_limits_commands(self):
        """Wheel ticks are blocked before 60ms, independently of movement."""
        self.state.scroll_mode_active = True
        move = lambda y, now: gestures.detect_vertical_movement(self.state, y, now)
        move(0.5, 0.0)
        self.assertEqual(move(0.46, 0.1), 1)
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 0.1), 1)
        self.assertEqual(move(0.42, 0.159), 1)
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 0.159), 0)
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 0.161), 1)

    def test_stationary_finger_continues_both_directions(self):
        """An initial movement is sufficient for repeated OS wheel commands."""
        for direction in (-1, 1):
            state = gestures.ScrollState(scroll_mode_active=True)
            with patch.object(gestures.pyautogui, "scroll") as wheel:
                gestures.detect_vertical_movement(state, 0.5, 0)
                self.assertEqual(gestures.continuous_scroll_direction(state, 0), 0)
                y = 0.5 - direction * 0.03
                for now in (0.1, 0.17, 0.24, 0.31):
                    gestures.detect_vertical_movement(state, y, now)
                    command = gestures.continuous_scroll_direction(state, now)
                    self.assertEqual(command, direction)
                    gestures.scroll(command, state.scroll_amount)
                self.assertEqual(wheel.call_count, 4)
                wheel.assert_called_with(direction * gestures.SCROLL_AMOUNT, _pause=False)

    def test_reversal_keeps_scrolling_until_fourth_frame(self):
        """The old direction persists during confirmation, then switches immediately."""
        for direction in (-1, 1):
            state = gestures.ScrollState(scroll_mode_active=True)
            gestures.detect_vertical_movement(state, 0.5, 0)
            y = 0.5 - direction * 0.1
            gestures.detect_vertical_movement(state, y, 0.1)
            self.assertEqual(gestures.continuous_scroll_direction(state, 0.1), direction)
            for index in range(1, 4):
                now = 0.1 + index * 0.07
                gestures.detect_vertical_movement(state, y + direction * index * 0.08, now)
                self.assertEqual(state.reverse_frame_count, index)
                self.assertEqual(gestures.continuous_scroll_direction(state, now), direction)
            # Only 1ms after the last wheel tick: confirmed reversal must not wait.
            gestures.detect_vertical_movement(state, y + direction * 0.32, 0.311)
            self.assertEqual(gestures.continuous_scroll_direction(state, 0.311), -direction)
            self.assertEqual(state.reverse_frame_count, 0)

    def test_stationary_frame_interrupts_reversal_not_scrolling(self):
        state = gestures.ScrollState(scroll_mode_active=True)
        for y, now in ((0.5, 0), (0.6, 0.1), (0.52, 0.2)):
            gestures.detect_vertical_movement(state, y, now)
        self.assertEqual(state.reverse_frame_count, 1)
        gestures.detect_vertical_movement(state, 0.52, 0.3)
        self.assertEqual(state.reverse_frame_count, 0)
        self.assertEqual(gestures.continuous_scroll_direction(state, 0.3), -1)

    def test_tracking_loss_and_fist_stop_continuous_ticks(self):
        self.activate()
        self.state.current_scroll_direction = "DOWN"
        ready = gestures.update_scroll_mode(self.state, None, 3)
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 3, ready), 0)
        for frame in range(gestures.HAND_LOSS_RESET_FRAMES):
            gestures.update_scroll_mode(self.state, None, 3 + frame * 0.1)
        self.assertIsNone(self.state.current_scroll_direction)
        self.assertTrue(self.show(1, now=4))
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 4), 0)
        self.state.current_scroll_direction = "UP"
        ready = self.show(0, now=4.1)
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 4.1, ready), 0)
        self.show(0, frames=2, now=4.2)
        self.assertFalse(self.state.scroll_mode_active)
        self.assertIsNone(self.state.current_scroll_direction)
        self.assertEqual(gestures.continuous_scroll_direction(self.state, 5), 0)

    def test_missing_hand_resets_baseline_and_confirmation(self):
        """Reacquisition must not scroll because the hand reappeared elsewhere."""
        self.activate()
        gestures.detect_vertical_movement(self.state, 0.5, 0)
        self.assertFalse(gestures.update_scroll_mode(self.state, None, 0.5))
        self.assertEqual(self.state.gesture_frames, 0)
        self.assertTrue(self.state.scroll_mode_active)
        self.assertTrue(self.show(2))
        self.assertEqual(gestures.detect_vertical_movement(self.state, 0.1, 1), 0)

    def test_activation_requires_two_seconds_and_resets_on_interruption(self):
        """Every non-open pose and missing hand must restart the activation hold."""
        for interruption in (0, 1, 2, 3, 4, None):
            self.state = gestures.ScrollState()
            self.show(5, now=10.0)
            self.show(5, now=11.99)
            self.assertFalse(self.state.scroll_mode_active)
            if interruption is None:
                gestures.update_scroll_mode(self.state, None, 12.0)
            else:
                self.show(interruption, now=12.0)
            self.assertIsNone(self.state.activation_started_at)
            self.show(5, now=13.0)
            self.show(5, now=14.99)
            self.assertFalse(self.state.scroll_mode_active)
            self.show(5, now=15.0)
            self.assertTrue(self.state.scroll_mode_active)
            self.assertIsNone(self.state.previous_y)

    def test_partial_hand_poses_keep_scrolling_active(self):
        """Partial poses retain mode; a fist requires a new full activation hold."""
        self.activate()
        for fingers in (1, 2, 3, 4, 5):
            self.assertTrue(self.show(fingers, now=3.0))
            self.assertTrue(self.state.scroll_mode_active)
        self.show(0, frames=3, now=4.0)
        self.assertFalse(self.state.scroll_mode_active)
        self.assertIsNone(self.state.activation_started_at)
        self.show(5, now=5.0)
        self.assertFalse(self.state.scroll_mode_active)
        self.show(5, now=7.0)
        self.assertTrue(self.state.scroll_mode_active)


if __name__ == "__main__":
    # Also allow running this file directly from the IDE.
    unittest.main()
