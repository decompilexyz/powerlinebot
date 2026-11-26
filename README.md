# Powerline.io temporal video policy

`train.py` fine-tunes a pretrained 3D ResNet-18 (`r3d_18`) on short video clips.
Unlike the earlier frame-only model, it processes 16 ordered frames over a
1.5-second window and predicts the arrow keys recorded at the end of that
window. This is supervised behavior cloning: it imitates demonstrated controls
and does not learn a reward for collecting food.

Gameplay pixels alone do not reveal which action was correct. Training therefore
requires an `actions.csv` with timestamps and arrow-key labels. Existing MP4s
without labels cannot directly train a playable policy; use `record_gameplay.py`
to create synchronized video and labels, or annotate your old videos.

## Install

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The first training run downloads the pretrained R3D-18 Kinetics weights. To
use an accelerator, choose the device explicitly: `mps` on Apple Silicon,
`cuda` on an NVIDIA machine, or `cpu`. `auto` prefers CUDA, then MPS, then CPU.
Install matching PyTorch and torchvision builds for your platform.

## Record video with action labels

Put the game fullscreen on the monitor you want to record, then run:

```bash
python record_gameplay.py --output-dir recordings/run-01
```

It records screen frames and currently held arrow keys at 10 FPS. Press `Esc`
to stop. Each run creates `gameplay.mp4` and `actions.csv`; choose a new output
directory for every run. On macOS, grant Screen Recording and Input Monitoring
permissions to Terminal/Python if prompted.

To improve the demonstrations, intentionally steer toward food and include
different situations, turns, and neutral frames. The model learns the recorded
behavior; it cannot infer a food-collection objective from unlabeled footage.

## Train

Train from a recorded run using the pretrained 3D CNN:

```bash
python train.py \
  --videos recordings/run-01 \
  --labels recordings/run-01/actions.csv \
  --output models/powerline_video.pt \
  --device mps
```

The CSV columns are `video,timestamp_seconds,actions`. `actions` can contain
multiple simultaneous keys separated by `|`, or be blank when no key is held.
The model uses that row's action as the label for a clip ending at its
timestamp. If a directory contains multiple videos, the validation split holds
out entire videos; for a single video it uses a chronological split.

Useful options:

- `--num-frames` (default 16) and `--clip-seconds` (default 1.5) control the
  temporal window.
- `--frame-width` and `--frame-height` default to 112, as expected by the
  pretrained video model.
- `--batch-size` defaults to 4 because 3D CNNs use more memory.
- `--epochs`, `--learning-rate`, `--validation-split`, and `--seed` adjust
  training.

If MPS or CUDA runs out of memory, lower `--batch-size`, for example to 2.
The saved checkpoint records the architecture, action labels, clip length, and
normalization needed by `play.py`.

## Test the trained policy

Keep the game fullscreen on the same monitor used for recording. Preview first;
this does not send keys:

```bash
python play.py --model models/powerline_video.pt --device mps
```

The runner collects a rolling clip before making its first prediction. Press
`Esc` to stop. If predictions look plausible, test live control briefly in
practice mode:

```bash
python play.py --model models/powerline_video.pt --device mps --play
```

Live control is opt-in. The model only imitates its demonstrations, so more
varied, intentionally food-seeking examples are needed for a better policy.

## Experimental food-targeting heuristic

The optional `--food-seeking` mode does not use a trained model. It looks for
small, saturated, round neon blobs and points toward the nearest one from the
screen center:

```bash
python play.py --food-seeking
```

This is preview-only unless `--play` is supplied. The detector may mistake
player trails or other effects for food, and it does not avoid walls or other
players. Inspect the preview before enabling live control.
