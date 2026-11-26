import unittest

from play import detect_food_target, food_target_action, select_actions


class SelectActionsTests(unittest.TestCase):
    def test_selects_labels_at_or_above_threshold(self):
        self.assertEqual(
            select_actions(["left", "right", "up"], [0.8, 0.2, 0.5], 0.5),
            ("left", "up"),
        )

    def test_rejects_mismatched_labels_and_probabilities(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            select_actions(["left"], [0.4, 0.6], 0.5)

    def test_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "range"):
            select_actions(["left"], [0.4], 1.1)


class FoodTargetActionTests(unittest.TestCase):
    def test_selects_axis_toward_target(self):
        self.assertEqual(food_target_action((10, 40), 100, 100), ("left",))
        self.assertEqual(food_target_action((80, 90), 100, 100), ("down",))
        self.assertEqual(food_target_action((60, 10), 100, 100), ("up",))

    def test_no_target_does_not_select_an_action(self):
        self.assertEqual(food_target_action(None, 100, 100), ())

    def test_rejects_invalid_frame_dimensions(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            food_target_action((10, 10), 0, 100)

    def test_detects_nearest_saturated_round_blob(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are required for image detection")

        frame = np.zeros((200, 300, 4), dtype=np.uint8)
        cv2.circle(frame, (40, 100), 6, (0, 255, 255, 255), -1)
        cv2.circle(frame, (220, 130), 6, (0, 255, 255, 255), -1)
        cv2.rectangle(frame, (0, 0), (299, 2), (255, 255, 255, 255), -1)

        self.assertEqual(detect_food_target(frame, cv2), (220, 130))


if __name__ == "__main__":
    unittest.main()
