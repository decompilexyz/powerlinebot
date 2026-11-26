#!/usr/bin/env python3
"""Record gameplay frames with synchronized arrow-key labels."""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path
from typing import Sequence


ARROW_ACTIONS = frozenset(("left", "right", "up", "down"))


def format_actions(pressed: set[str]) -> str:
    """Format the currently held arrow keys for the training CSV."""
    return "|".join(sorted(pressed & ARROW_ACTIONS))


def _load_recording_dependencies():
    try:
        import cv2
        import mss
        import numpy as np
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError(
            "Recording requires the packages in requirements.txt. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc
    return cv2, mss, np, keyboard


def record_gameplay(output_dir: Path, monitor_index: int, fps: float) -> tuple[Path, Path]:
    if fps <= 0:
        raise ValueError("--fps must be greater than 0")

    cv2, mss, np, keyboard = _load_recording_dependencies()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "gameplay.mp4"
    labels_path = output_dir / "actions.csv"
    if video_path.exists() or labels_path.exists():
        raise FileExistsError(
            f"{output_dir} already contains a recording; choose a new --output-dir"
        )

    pressed: set[str] = set()
    pressed_lock = threading.Lock()
    stop_event = threading.Event()
    key_actions = {
        keyboard.Key.left: "left",
        keyboard.Key.right: "right",
        keyboard.Key.up: "up",
        keyboard.Key.down: "down",
    }

    def on_press(key):
        if key == keyboard.Key.esc:
            stop_event.set()
            return False
        action = key_actions.get(key)
        if action:
            with pressed_lock:
                pressed.add(action)

    def on_release(key):
        action = key_actions.get(key)
        if action:
            with pressed_lock:
                pressed.discard(action)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        print("Switch to the fullscreen game. Recording starts in 3 seconds.")
        for remaining in (3, 2, 1):
            print(f"{remaining}...")
            if stop_event.wait(1):
                raise RuntimeError("Recording cancelled before it started")

        with pressed_lock:
            pressed.clear()

        with mss.mss() as screen:
            if monitor_index < 0 or monitor_index >= len(screen.monitors):
                raise ValueError(
                    f"Monitor {monitor_index} is unavailable; "
                    f"choose a value from 0 to {len(screen.monitors) - 1}"
                )
            region = dict(screen.monitors[monitor_index])
            region["width"] -= region["width"] % 2
            region["height"] -= region["height"] % 2
            if region["width"] < 2 or region["height"] < 2:
                raise ValueError("Selected monitor is too small to record")

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (region["width"], region["height"]),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Could not create video file: {video_path}")

            rows = []
            frame_index = 0
            start_time = time.monotonic()
            print("Recording. Press Esc to stop.")
            try:
                while not stop_event.is_set():
                    deadline = start_time + frame_index / fps
                    if stop_event.wait(max(0, deadline - time.monotonic())):
                        break

                    frame = np.asarray(screen.grab(region))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    writer.write(frame)

                    with pressed_lock:
                        actions = format_actions(pressed)
                    rows.append(
                        {
                            "video": video_path.name,
                            "timestamp_seconds": f"{frame_index / fps:.3f}",
                            "actions": actions,
                        }
                    )
                    frame_index += 1
            finally:
                writer.release()

        if frame_index == 0:
            raise RuntimeError("No frames were recorded")
        with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ("video", "timestamp_seconds", "actions")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        labeled_frames = sum(bool(row["actions"]) for row in rows)
        if not labeled_frames:
            print("Warning: no arrow-key presses were captured; this run cannot train a policy.")
        print(f"Saved {frame_index} frames to {video_path}")
        print(f"Saved synchronized labels to {labels_path}")
        return video_path, labels_path
    finally:
        listener.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record gameplay video and synchronized arrow-key labels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for gameplay.mp4 and actions.csv",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Screen to record (1 is the primary display; default: 1)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10,
        help="Frames and labels per second (default: 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        record_gameplay(args.output_dir, args.monitor, args.fps)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
