import csv
import tempfile
import unittest
from pathlib import Path

from train import (
    Annotation,
    _conv_feature_size,
    action_vocabulary,
    load_annotations,
    _build_video_policy,
    sample_clip_indices,
    split_annotation_indices,
)


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


class VideoClipTests(unittest.TestCase):
    def test_samples_ordered_frames_over_the_requested_temporal_window(self):
        indices = sample_clip_indices(100, 10, 16, 1.5)

        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 85)
        self.assertEqual(indices[-1], 100)
        self.assertEqual(indices, sorted(indices))

    def test_pads_early_clips_by_repeating_available_first_frame(self):
        self.assertEqual(sample_clip_indices(0, 10, 4, 1.5), [0, 0, 0, 0])

    def test_rejects_invalid_clip_configuration(self):
        with self.assertRaisesRegex(ValueError, "video_fps"):
            sample_clip_indices(10, 0, 16, 1.5)

    def test_single_video_validation_split_is_chronological(self):
        video = Path("/videos/round.mp4")
        annotations = [
            Annotation(video, timestamp, ("left",))
            for timestamp in (3.0, 0.0, 2.0, 1.0, 4.0)
        ]

        train_indices, validation_indices = split_annotation_indices(
            annotations,
            0.4,
            42,
        )

        self.assertEqual(
            [annotations[index].timestamp_seconds for index in train_indices],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(
            [annotations[index].timestamp_seconds for index in validation_indices],
            [3.0, 4.0],
        )

    def test_single_video_split_purges_overlapping_clip_windows(self):
        video = Path("/videos/round.mp4")
        annotations = [
            Annotation(video, timestamp, ("left",))
            for timestamp in range(10)
        ]

        train_indices, validation_indices = split_annotation_indices(
            annotations,
            0.2,
            42,
            clip_seconds=2,
        )

        self.assertEqual(
            [annotations[index].timestamp_seconds for index in train_indices],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        self.assertEqual(
            [annotations[index].timestamp_seconds for index in validation_indices],
            [8.0, 9.0],
        )

    def test_multi_video_validation_holds_out_whole_videos(self):
        first_video = Path("/videos/round-01.mp4")
        second_video = Path("/videos/round-02.mp4")
        annotations = [
            Annotation(first_video, 0.0, ("left",)),
            Annotation(first_video, 1.0, ("left",)),
            Annotation(second_video, 0.0, ("right",)),
            Annotation(second_video, 1.0, ("right",)),
        ]

        train_indices, validation_indices = split_annotation_indices(
            annotations,
            0.5,
            42,
        )

        self.assertTrue(train_indices)
        self.assertTrue(validation_indices)
        self.assertEqual(
            {annotations[index].video for index in train_indices}
            & {annotations[index].video for index in validation_indices},
            set(),
        )


class VideoModelTests(unittest.TestCase):
    def test_r3d_model_accepts_temporal_clips_and_emits_action_logits(self):
        try:
            import torch
            import torchvision
        except ImportError:
            self.skipTest("PyTorch and torchvision are required for the video model")

        model = _build_video_policy(torch, action_count=4).eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, 16, 112, 112))

        self.assertEqual(tuple(logits.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
