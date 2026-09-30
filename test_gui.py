"""Window lifecycle tests without opening a real camera window."""

import unittest
from unittest.mock import patch

import gui


class WindowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.multiple(gui, _window_created=False, _exit_requested=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_close_button_does_not_recreate_window(self):
        with patch.object(gui, 'draw_welcome'), patch.object(gui.cv2, 'imshow') as show, \
                patch.object(gui.cv2, 'waitKey', return_value=-1), \
                patch.object(gui.cv2, 'getWindowProperty', side_effect=[1, 0]):
            self.assertFalse(gui.show_frame(None))
            self.assertTrue(gui.show_frame(None))
            self.assertTrue(gui.show_frame(None))
            show.assert_called_once()

    def test_destroyed_window_error_means_exit(self):
        gui._window_created = True
        with patch.object(gui.cv2, 'getWindowProperty', side_effect=gui.cv2.error()), \
                patch.object(gui.cv2, 'imshow') as show:
            self.assertTrue(gui.show_frame(None))
            show.assert_not_called()

    def test_q_prevents_further_display(self):
        with patch.object(gui, 'draw_welcome'), patch.object(gui.cv2, 'imshow') as show, \
                patch.object(gui.cv2, 'waitKey', return_value=ord('q')):
            self.assertTrue(gui.show_frame(None))
            self.assertTrue(gui.show_frame(None))
            show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
