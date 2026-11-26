import unittest

from play import select_actions


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


if __name__ == "__main__":
    unittest.main()
