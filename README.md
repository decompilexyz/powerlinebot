# Powerline.io video behavior cloning

`train.py` learns a visual action policy from gameplay frames paired with the
controls that were active in those frames. It does not infer controller inputs
from pixels: each training example must be annotated in a CSV file. `play.py`
can preview the trained model and, when explicitly enabled, send arrow keys to
the game.

## Install

Use Python 3.10 or newer. Install the training and video-decoding dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch chooses the available accelerator automatically by default (CUDA, then
Apple MPS, then CPU). Install the appropriate PyTorch build for your machine if
the default package does not provide your desired accelerator.

## Annotate gameplay

Create a CSV with one labeled sample per row:

```csv
video,timestamp_seconds,actions
round-01.mp4,3.2,left
round-01.mp4,4.0,left|boost
round-02.mp4,8.5,
```

- `video` is a path relative to the `--videos` directory. When `--videos`
  points to one file, use that file's basename.
- `timestamp_seconds` is the frame time measured from the beginning of that
  video.
- `actions` contains the exact action names you want the model to predict.
  Separate simultaneously active controls with `|`; leave it blank when no
  control is active. The action vocabulary is learned from the labels and is
  not hard-coded by the script.

The sample labels above are illustrative only; provide the action names and
timestamps that match your own recordings and controller mapping.

## Record a new training session (keyboard controls)

The existing MP4s do not contain arrow-key events, so the script cannot recover
those labels from the old recordings. To make a synchronized video and labels
file automatically, install the dependencies above, put the game in fullscreen,
then run:

```bash
python record_gameplay.py --output-dir recordings/run-01
```

The recorder waits three seconds, captures the primary display at 10 frames per
second, and records only which arrow keys are held at each frame. Press `Esc`
to stop. It creates `gameplay.mp4` and `actions.csv` in the output directory;
choose a different directory for each run. Close unrelated private windows
before recording. On macOS, grant Screen Recording and Input Monitoring
permissions to the terminal or Python if the system requests them, then restart
the terminal.

Train from that recorded run with:

```bash
python train.py \
  --videos recordings/run-01 \
  --labels recordings/run-01/actions.csv \
  --output models/powerline_behavior.pt
```

## Train

Point `--videos` at a single MP4 or a directory of recordings:

```bash
python train.py \
  --videos "$HOME/Videos/powerline" \
  --labels "$HOME/Videos/powerline/actions.csv" \
  --output models/powerline_behavior.pt
```

Useful options include `--epochs`, `--batch-size`, `--validation-split`,
`--frame-width`, `--frame-height`, `--seed`, and `--device auto|cpu|cuda|mps`.
The checkpoint contains model weights, action labels, and image preprocessing
dimensions. Each action has an independent sigmoid probability; a consuming
controller can interpret probabilities at its chosen threshold (0.5 is a
common starting point).

This is supervised behavior cloning, not reinforcement learning. Its quality
depends on accurate, representative labels; it does not discover better play
than the demonstrations.

## Test the trained model

Put the game in fullscreen on the same display used for recording. First run a
prediction preview; this does not send any keys:

```bash
python play.py --model models/powerline_behavior.pt
```

Press `Esc` to stop. If the predictions look reasonable, try a short session in
practice mode with explicit key control enabled:

```bash
python play.py --model models/powerline_behavior.pt --play
```

The runner only sends arrow keys, and `Esc` stops it and releases held keys.
On macOS, allow Screen Recording and Input Monitoring if prompted. Start with a
short session: a small training recording is only a smoke test, and the model
may not play well until it has more varied demonstrations.

### Experimental food targeting

The trained policy imitates recorded key presses; it does not identify food or
optimize for collecting it. For a rough food-targeting experiment, `play.py`
also has a heuristic mode that looks for small, bright, saturated, round blobs
and points toward the nearest one relative to the center of the captured
screen. It does not use or require a model:

```bash
python play.py --food-seeking
```

This mode is preview-only unless `--play` is explicitly supplied. Check its
suggested targets before enabling live control:

```bash
python play.py --food-seeking --play
```

The detector is an untrained visual heuristic, not a game-aware planner. Neon
player trails or other effects may be mistaken for food, and the controller
does not avoid walls or other players. Use it only in a supervised practice
session; if the preview targets the wrong objects, do not enable `--play`.
