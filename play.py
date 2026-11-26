#!/usr/bin/env python3
"""Preview a trained policy or explicitly let it control arrow keys."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from record_gameplay import ARROW_ACTIONS
from train import (
    _build_policy,
    _conv_feature_size,
    _select_device,
)


def select_actions(
    action_labels: Sequence[str],
    probabilities: Sequence[float],
    threshold: float,
) -> tuple[str, ...]:
    if len(action_labels) != len(probabilities):
        raise ValueError("Action labels and probabilities must have the same length")
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be in the range [0, 1]")
    return tuple(
        label
        for label, probability in zip(action_labels, probabilities)
        if probability >= threshold
    )


def food_target_action(
    target: tuple[int, int] | None,
    frame_width: int,
    frame_height: int,
) -> tuple[str, ...]:
    if target is None:
        return ()
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be greater than zero")

    dx = target[0] - frame_width / 2
    dy = target[1] - frame_height / 2
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left",)
    return ("down" if dy > 0 else "up",)


def detect_food_target(frame, cv2) -> tuple[int, int] | None:
    """Find the nearest round, saturated neon object away from the screen center."""
    height, width = frame.shape[:2]
    minimum_dimension = min(width, height)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 150), (179, 255, 255))
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    center_x, center_y = width / 2, height / 2
    min_diameter = max(4, minimum_dimension * 0.005)
    max_diameter = minimum_dimension * 0.08
    center_exclusion = minimum_dimension * 0.025
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        diameter = max(box_width, box_height)
        aspect_ratio = box_width / box_height
        circularity = 4 * 3.141592653589793 * area / (perimeter * perimeter)
        if (
            diameter < min_diameter
            or diameter > max_diameter
            or not 0.65 <= aspect_ratio <= 1.55
            or circularity < 0.35
        ):
            continue

        point = (x + box_width // 2, y + box_height // 2)
        distance_squared = (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2
        if distance_squared <= center_exclusion**2:
            continue
        candidates.append((distance_squared, point))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _build_legacy_policy(torch, action_count: int):
    """Build the original architecture for checkpoints trained before the MPS fix."""
    nn = torch.nn

    class LegacyVisionPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, action_count),
            )

        def forward(self, observations):
            return self.classifier(self.features(observations))

    return LegacyVisionPolicy()


def _load_policy(torch, checkpoint_path: Path, device):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"Model checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    required = ("model_state_dict", "action_labels", "frame_width", "frame_height")
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in required):
        raise ValueError(f"Unsupported model checkpoint: {checkpoint_path}")

    action_labels = checkpoint["action_labels"]
    unsupported = set(action_labels).difference(ARROW_ACTIONS)
    if unsupported:
        raise ValueError(
            "This runner only supports arrow-key actions; checkpoint contains: "
            + ", ".join(sorted(unsupported))
        )

    state = checkpoint["model_state_dict"]
    classifier_weight = state.get("classifier.0.weight")
    if classifier_weight is None:
        raise ValueError("Checkpoint is missing its policy classifier weights")
    legacy_input_size = 64 * 4 * 4
    if classifier_weight.shape[1] == legacy_input_size:
        if device.type == "mps":
            raise ValueError(
                "This checkpoint uses the older MPS-incompatible pooling layer. "
                "Retry with --device cpu, or retrain after updating the code."
            )
        model = _build_legacy_policy(torch, len(action_labels))
    else:
        expected_width = _conv_feature_size(checkpoint["frame_width"])
        expected_height = _conv_feature_size(checkpoint["frame_height"])
        expected_input_size = 64 * expected_width * expected_height
        if classifier_weight.shape[1] != expected_input_size:
            raise ValueError("Checkpoint architecture does not match its saved frame size")
        model = _build_policy(
            torch,
            len(action_labels),
            checkpoint["frame_width"],
            checkpoint["frame_height"],
        )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint, action_labels


def run(args: argparse.Namespace) -> None:
    try:
        import cv2
        import mss
        import numpy as np
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError(
            "Testing requires the packages in requirements.txt. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc

    if args.food_seeking:
        model = checkpoint = action_labels = device = None
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Model testing requires PyTorch. "
                "Install it with: python -m pip install -r requirements.txt"
            ) from exc
        device = _select_device(torch, args.device)
        model, checkpoint, action_labels = _load_policy(torch, args.model, device)

    key_by_action = {
        "left": keyboard.Key.left,
        "right": keyboard.Key.right,
        "up": keyboard.Key.up,
        "down": keyboard.Key.down,
    }

    stop_event = threading.Event()

    def on_press(key):
        if key == keyboard.Key.esc:
            stop_event.set()
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    controller = keyboard.Controller() if args.play else None
    held_keys = set()
    if checkpoint is not None:
        frame_width = checkpoint["frame_width"]
        frame_height = checkpoint["frame_height"]
    frame_interval = 1.0 / args.fps
    last_print = 0.0

    try:
        print("Put the game in fullscreen on the selected monitor.")
        if args.food_seeking:
            print(
                "Experimental food targeting: aiming at the nearest detected "
                "neon blob from screen center."
            )
        if args.play:
            print("LIVE CONTROL ENABLED. Press Esc to stop and release all keys.")
        else:
            print("Preview only: no game keys will be pressed. Press Esc to stop.")
        for remaining in (3, 2, 1):
            print(f"{remaining}...")
            if stop_event.wait(1):
                return

        with mss.MSS() as screen:
            if args.monitor < 0 or args.monitor >= len(screen.monitors):
                raise ValueError(
                    f"Monitor {args.monitor} is unavailable; "
                    f"choose a value from 0 to {len(screen.monitors) - 1}"
                )
            monitor = screen.monitors[args.monitor]
            deadline = time.monotonic()
            print("Running. Press Esc to stop.")
            while not stop_event.is_set():
                raw_frame = np.asarray(screen.grab(monitor))
                target = None
                if args.food_seeking:
                    target = detect_food_target(raw_frame, cv2)
                    actions = set(
                        food_target_action(
                            target,
                            raw_frame.shape[1],
                            raw_frame.shape[0],
                        )
                    )
                else:
                    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2RGB)
                    rgb_frame = cv2.resize(
                        rgb_frame,
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    image = np.ascontiguousarray(np.transpose(rgb_frame, (2, 0, 1)))
                    observations = torch.from_numpy(image).unsqueeze(0)
                    observations = observations.to(
                        device=device,
                        dtype=torch.float32,
                    ) / 255.0

                    with torch.no_grad():
                        probabilities = torch.sigmoid(model(observations))[0].cpu().tolist()
                    actions = set(
                        select_actions(action_labels, probabilities, args.threshold)
                    )

                if controller is not None:
                    desired_keys = {key_by_action[action] for action in actions}
                    for key in held_keys - desired_keys:
                        controller.release(key)
                    for key in desired_keys - held_keys:
                        controller.press(key)
                    held_keys = desired_keys

                now = time.monotonic()
                if now - last_print >= 0.5:
                    if args.food_seeking:
                        action_text = "+".join(sorted(actions)) or "no target"
                        target_text = f" target={target}" if target is not None else ""
                        mode = "FOOD PLAY" if args.play else "FOOD PREVIEW"
                        status = f"{mode}: {action_text}{target_text}"
                    else:
                        action_text = "+".join(sorted(actions)) or "none"
                        mode = "PLAY" if args.play else "PREVIEW"
                        status = f"{mode} predicted: {action_text}"
                    print(f"\r{status:<64}", end="", flush=True)
                    last_print = now

                deadline += frame_interval
                stop_event.wait(max(0, deadline - time.monotonic()))
            print()
    finally:
        if controller is not None:
            for key in held_keys:
                controller.release(key)
        listener.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview a trained arrow-key policy, use experimental food targeting, "
            "or explicitly let either control the game."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/powerline_behavior.pt"),
        help="Trained checkpoint (default: models/powerline_behavior.pt)",
    )
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--food-seeking",
        action="store_true",
        help=(
            "Use experimental neon-blob targeting instead of the trained model; "
            "preview only unless --play is also given"
        ),
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Send selected arrow keys to the game; default is preview-only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in the range [0, 1]")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
