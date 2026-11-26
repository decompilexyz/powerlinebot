import unittest

from record_gameplay import format_actions


class FormatActionsTests(unittest.TestCase):
    def test_formats_held_arrow_keys_in_stable_order(self):
        self.assertEqual(format_actions({"right", "left"}), "left|right")

    def test_ignores_non_arrow_keys_and_supports_no_key(self):
        self.assertEqual(format_actions({"space", "a"}), "")


if __name__ == "__main__":
    unittest.main()
