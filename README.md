# Powerline.io video behavior cloning

`train.py` learns a visual action policy from gameplay frames paired with the
controls that were active in those frames. It does not open a browser, play the
game, or infer controller inputs from pixels: each training example must be
annotated in a CSV file.

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
than the demonstrations or interact with Powerline.io on its own.
