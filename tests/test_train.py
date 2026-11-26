import csv
import tempfile
import unittest
from pathlib import Path

from train import _conv_feature_size, action_vocabulary, load_annotations


class LoadAnnotationsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        (self.videos / "round-01.mp4").write_bytes(b"test video placeholder")
        self.labels = self.root / "actions.csv"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_csv(self, rows, columns=("video", "timestamp_seconds", "actions")):
        with self.labels.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_reads_relative_videos_and_simultaneous_actions(self):
        self.write_csv(
            [
                {
                    "video": "round-01.mp4",
                    "timestamp_seconds": "3.25",
                    "actions": "turn-left|boost| turn-left ",
                },
                {
                    "video": "round-01.mp4",
                    "timestamp_seconds": "4",
                    "actions": "",
                },
            ]
        )

        annotations = load_annotations(self.labels, self.videos)

        self.assertEqual(annotations[0].video, (self.videos / "round-01.mp4").resolve())
        self.assertEqual(annotations[0].timestamp_seconds, 3.25)
        self.assertEqual(annotations[0].actions, ("boost", "turn-left"))
        self.assertEqual(annotations[1].actions, ())
        self.assertEqual(action_vocabulary(annotations), ["boost", "turn-left"])

    def test_accepts_selected_video_file_by_basename(self):
        video = self.videos / "round-01.mp4"
        self.write_csv(
            [{"video": video.name, "timestamp_seconds": "0", "actions": "turn"}]
        )

        annotations = load_annotations(self.labels, video)

        self.assertEqual(annotations[0].video, video.resolve())

    def test_rejects_negative_or_non_finite_timestamps(self):
        for timestamp in ("-1", "nan", "inf"):
            with self.subTest(timestamp=timestamp):
                self.write_csv(
                    [
                        {
                            "video": "round-01.mp4",
                            "timestamp_seconds": timestamp,
                            "actions": "turn",
                        }
                    ]
                )
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    load_annotations(self.labels, self.videos)

    def test_rejects_video_paths_outside_selected_directory(self):
        self.write_csv(
            [
                {
                    "video": "../outside.mp4",
                    "timestamp_seconds": "0",
                    "actions": "turn",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "outside the selected video directory"):
            load_annotations(self.labels, self.videos)

    def test_requires_an_action_label_somewhere_in_dataset(self):
        self.write_csv(
            [
                {
                    "video": "round-01.mp4",
                    "timestamp_seconds": "0",
                    "actions": "",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "at least one non-empty action label"):
            load_annotations(self.labels, self.videos)


class ConvFeatureSizeTests(unittest.TestCase):
    def test_computes_convolution_output_dimensions(self):
        self.assertEqual(_conv_feature_size(84), 7)
        self.assertEqual(_conv_feature_size(36), 1)

    def test_rejects_dimensions_too_small_for_network(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            _conv_feature_size(35)


if __name__ == "__main__":
    unittest.main()
