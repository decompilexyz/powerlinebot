#!/usr/bin/env python3
"""Train a Powerline.io behavior-cloning policy from annotated gameplay videos."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Annotation:
    video: Path
    timestamp_seconds: float
    actions: tuple[str, ...]


def _resolve_video(reference: str, video_source: Path, csv_directory: Path) -> Path:
    source = video_source.resolve()
    supplied = Path(reference).expanduser()

    if source.is_file():
        if supplied.is_absolute():
            resolved = supplied.resolve()
        elif supplied.name == source.name:
            resolved = source
        else:
            resolved = (csv_directory / supplied).resolve()
        if resolved != source:
            raise ValueError(
                f"CSV video {reference!r} does not refer to the selected video "
                f"{str(source)!r}"
            )
        return source

    if supplied.is_absolute():
        resolved = supplied.resolve()
    else:
        resolved = (source / supplied).resolve()

    try:
        resolved.relative_to(source)
    except ValueError as exc:
        raise ValueError(
            f"CSV video {reference!r} is outside the selected video directory "
            f"{str(source)!r}"
        ) from exc

    if not resolved.is_file():
        raise ValueError(f"CSV video file does not exist: {resolved}")
    return resolved


def load_annotations(labels_path: Path, video_source: Path) -> list[Annotation]:
    """Read timestamped multi-label examples from the user's CSV file."""
    labels_path = labels_path.expanduser().resolve()
    video_source = video_source.expanduser().resolve()
    if not labels_path.is_file():
        raise ValueError(f"Labels CSV does not exist: {labels_path}")
    if not video_source.exists():
        raise ValueError(f"Video path does not exist: {video_source}")
    if not (video_source.is_file() or video_source.is_dir()):
        raise ValueError(f"Video path is not a file or directory: {video_source}")

    annotations: list[Annotation] = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"video", "timestamp_seconds", "actions"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Labels CSV is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            reference = (row.get("video") or "").strip()
            if not reference:
                raise ValueError(f"Labels CSV row {row_number}: video cannot be empty")
            try:
                timestamp = float(row["timestamp_seconds"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Labels CSV row {row_number}: timestamp_seconds must be a number"
                ) from exc
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(
                    f"Labels CSV row {row_number}: timestamp_seconds must be finite "
                    "and non-negative"
                )

            action_text = (row.get("actions") or "").strip()
            actions = tuple(
                sorted({item.strip() for item in action_text.split("|") if item.strip()})
            )
            video = _resolve_video(reference, video_source, labels_path.parent)
            annotations.append(Annotation(video, timestamp, actions))

    if not annotations:
        raise ValueError(f"Labels CSV contains no examples: {labels_path}")
    if not any(annotation.actions for annotation in annotations):
        raise ValueError("Labels CSV must contain at least one non-empty action label")
    return annotations


def action_vocabulary(annotations: Sequence[Annotation]) -> list[str]:
    return sorted({action for item in annotations for action in item.actions})


def _conv_feature_size(size: int) -> int:
    for kernel_size, stride in ((8, 4), (4, 2), (3, 1)):
        size = (size - kernel_size) // stride + 1
        if size < 1:
            raise ValueError("Frame dimensions are too small for the CNN")
    return size


def _load_training_dependencies():
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the packages in requirements.txt. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc
    return cv2, np, torch


def _build_policy(torch, action_count: int, frame_width: int, frame_height: int):
    nn = torch.nn
    feature_width = _conv_feature_size(frame_width)
    feature_height = _conv_feature_size(frame_height)

    class VisionPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * feature_width * feature_height, 256),
                nn.ReLU(),
                nn.Linear(256, action_count),
            )

        def forward(self, observations):
            return self.classifier(self.features(observations))

    return VisionPolicy()


def _prepare_dataset(annotations, actions, frame_width, frame_height, cv2, np, torch):
    action_indices = {action: index for index, action in enumerate(actions)}
    video_info = {}

    for annotation in annotations:
        if annotation.video in video_info:
            continue
        capture = cv2.VideoCapture(str(annotation.video))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {annotation.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError(f"Video has invalid frame metadata: {annotation.video}")
        video_info[annotation.video] = (fps, frame_count)

    for annotation in annotations:
        fps, frame_count = video_info[annotation.video]
        frame_index = int(round(annotation.timestamp_seconds * fps))
        if frame_index >= frame_count:
            duration = frame_count / fps
            raise ValueError(
                f"Timestamp {annotation.timestamp_seconds:g}s in "
                f"{annotation.video.name!r} is outside the video "
                f"(duration {duration:.3f}s)"
            )

    class VideoActionDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.captures = {}

        def __len__(self):
            return len(annotations)

        def __getitem__(self, index):
            annotation = annotations[index]
            capture = self.captures.get(annotation.video)
            if capture is None:
                capture = cv2.VideoCapture(str(annotation.video))
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not reopen video: {annotation.video}")
                self.captures[annotation.video] = capture

            fps, _ = video_info[annotation.video]
            frame_index = int(round(annotation.timestamp_seconds * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {annotation.video}"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame,
                (frame_width, frame_height),
                interpolation=cv2.INTER_AREA,
            )
            image = np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))
            target = np.zeros(len(actions), dtype=np.float32)
            for action in annotation.actions:
                target[action_indices[action]] = 1.0
            return torch.from_numpy(image), torch.from_numpy(target)

        def close(self):
            for capture in self.captures.values():
                capture.release()
            self.captures.clear()

    return VideoActionDataset()


def _select_device(torch, requested: str):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.type == "mps" and (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> Path:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if not 0 <= args.validation_split < 1:
        raise ValueError("--validation-split must be in the range [0, 1)")
    if args.frame_width < 36 or args.frame_height < 36:
        raise ValueError("--frame-width and --frame-height must both be at least 36")

    annotations = load_annotations(args.labels, args.videos)
    actions = action_vocabulary(annotations)
    cv2, np, torch = _load_training_dependencies()
    torch.manual_seed(args.seed)

    device = _select_device(torch, args.device)
    dataset = _prepare_dataset(
        annotations,
        actions,
        args.frame_width,
        args.frame_height,
        cv2,
        np,
        torch,
    )

    try:
        validation_count = int(len(dataset) * args.validation_split)
        if args.validation_split > 0 and validation_count == 0:
            minimum_examples = math.ceil(1 / args.validation_split)
            raise ValueError(
                f"At least {minimum_examples} labeled examples are required for "
                f"--validation-split {args.validation_split:g}"
            )
        train_count = len(dataset) - validation_count
        generator = torch.Generator().manual_seed(args.seed)
        if validation_count:
            train_dataset, validation_dataset = torch.utils.data.random_split(
                dataset, [train_count, validation_count], generator=generator
            )
        else:
            train_dataset, validation_dataset = dataset, None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        validation_loader = (
            torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            if validation_dataset is not None
            else None
        )

        model = _build_policy(
            torch,
            len(actions),
            args.frame_width,
            args.frame_height,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_function = torch.nn.BCEWithLogitsLoss()
        best_validation_loss = math.inf

        def evaluate(loader):
            model.eval()
            total_loss = 0.0
            total_examples = 0
            with torch.no_grad():
                for images, targets in loader:
                    images = images.to(device=device, dtype=torch.float32) / 255.0
                    targets = targets.to(device=device)
                    loss = loss_function(model(images), targets)
                    total_loss += loss.item() * len(images)
                    total_examples += len(images)
            return total_loss / total_examples

        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for images, targets in train_loader:
                images = images.to(device=device, dtype=torch.float32) / 255.0
                targets = targets.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(images), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(images)
                total_examples += len(images)

            train_loss = total_loss / total_examples
            if validation_loader is not None:
                validation_loss = evaluate(validation_loader)
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"train_loss={train_loss:.4f} "
                    f"validation_loss={validation_loss:.4f}"
                )
                should_save = validation_loss < best_validation_loss
                best_validation_loss = min(best_validation_loss, validation_loss)
            else:
                print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}")
                should_save = epoch == args.epochs

            if should_save:
                torch.save(
                    {
                        "format_version": 1,
                        "model_state_dict": model.state_dict(),
                        "action_labels": actions,
                        "frame_width": args.frame_width,
                        "frame_height": args.frame_height,
                        "input_color_space": "RGB",
                        "input_layout": "NCHW",
                    },
                    output_path,
                )
    finally:
        dataset.close()

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a multi-label Powerline.io policy from annotated MP4 frames."
    )
    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="A gameplay video file or a directory containing the videos",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV with columns: video,timestamp_seconds,actions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/powerline_behavior.pt"),
        help="Checkpoint path (default: models/powerline_behavior.pt)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--frame-width", type=int, default=84)
    parser.add_argument("--frame-height", type=int, default=84)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device (default: prefer CUDA, then Apple MPS, then CPU)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = train(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved behavior-cloning checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
