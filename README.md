# smolVLA-training

Post-training [SmolVLA](https://huggingface.co/lerobot/smolvla_base) into a **lane-following policy** for the DitecT Lab scaled autonomous car (E1).

The pipeline takes the car's raw driving recordings (ROS 2 bags + front camera video), converts them into a [LeRobot](https://github.com/huggingface/lerobot) dataset, and fine-tunes SmolVLA (450M) to map camera frames to steering and speed. The trained policy is meant to run on the car's Jetson.

## Status

| Step | Status |
|---|---|
| Convert raw runs to a LeRobot dataset | Done: 102 episodes, 148,371 frames (~82 min) from the AoE 2026 runs |
| Fine-tune SmolVLA on the dataset | In progress |
| Closed-loop evaluation on the track | Planned |
| Export for the Jetson (TensorRT) | Planned |

## Repository structure

```
.
├── convert_runs.py        # raw ROS 2 runs -> LeRobot dataset
├── conversion_guide.pdf   # write-up of the conversion (labeling sections are outdated)
├── lerobot/               # LeRobot, as a git submodule pinned to the version we use
└── data/                  # not in git, see "Data" below
    ├── raw/               # one folder per run (E1-1022-01 ... E1-1022-40)
    └── lerobot/aoe_lane/  # converted dataset written by convert_runs.py
```

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg (for video encoding and decoding)
- A Hugging Face account, needed only to push or pull datasets and models
- A CUDA GPU for training. Apple Silicon (MPS) works for testing but is slow.

## Setup

```bash
# 1. Clone with the LeRobot submodule
git clone --recurse-submodules https://github.com/jakeli00/smolVLA-training.git
cd smolVLA-training

# 2. Create the environment inside lerobot/ and install LeRobot with SmolVLA
cd lerobot
uv venv --python 3.12
uv pip install -e ".[smolvla,training]"
uv pip install rosbags opencv-python
cd ..

# 3. (Optional) Log in to Hugging Face to push or pull datasets
lerobot/.venv/bin/hf auth login
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init`.

## Data

The data is too large for git and is shared separately by the lab. Put it here:

```
data/raw/E1-1022-01/
├── video.ts      # front camera, 1280x960 @ 30 fps, mounted upside down
├── frames.csv    # wall-clock time of each video frame
├── meta.json     # car id, start/end time, steering trim, topics
└── bag/          # ROS 2 bag (.db3): /ackermann_cmd and /odom at ~50 Hz
```

Runs 08 and 16 have empty bags and run 12 has no video frames. The script skips all three automatically.

## Usage

### 1. Convert the raw runs

First, edit the two paths at the top of `convert_runs.py` (`raw` and `out`) to point at your copy of `data/`. Then run:

```bash
lerobot/.venv/bin/python convert_runs.py
```

This takes about 2 hours, almost all of it AV1 video encoding. The script refuses to run if the output folder already exists, so delete it first to rebuild.

What the script does for each run:

1. Matches every video frame to the steering/speed command at or just after it, and to the latest odometry reading. The largest gaps are about 20 ms for commands and 25 ms for odometry.
2. Splits the run into episodes, where each episode is a stretch with commanded speed above 0.05 m/s. Stops under 1 s stay inside an episode, and stretches under 2 s are dropped.
3. Rotates each frame 180 degrees, resizes it to 640x480 and writes it with:

| Feature | Shape | Contents |
|---|---|---|
| `observation.images.front` | 480x640x3 | front camera |
| `observation.state` | 2 | measured speed (m/s), previous steering (rad) |
| `action` | 2 | steering (rad), speed (m/s) |
| `task` | text | `"follow the lane"` (same for every frame) |

### 2. (Optional) Share the dataset on the Hub

```bash
lerobot/.venv/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
LeRobotDataset('jakeli00/ditect_aoe_lane', root='data/lerobot/aoe_lane').push_to_hub(private=True)"
```

### 3. Fine-tune SmolVLA

This is a starting command based on the [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla). Tune the batch size and steps for your GPU.

```bash
lerobot/.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=jakeli00/ditect_aoe_lane \
  --dataset.root=data/lerobot/aoe_lane \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla_lane \
  --job_name=smolvla_lane \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --trackio.enable=true
```

`--rename_map` maps our camera name onto the camera slot SmolVLA was pretrained with. Run `trackio show` to watch the training curves locally.

## Notes

- `/odom` reports speed with the opposite sign of the command, so the script negates it.
- Speed is capped at 0.5 m/s by the car's bridge, and steering trim is -0.12 on every run.
- The data includes intersections and turns, not just straight lane following. A single `"follow the lane"` task may be ambiguous at intersections.
- On first load, LeRobot may print `torchcodec` errors with newer ffmpeg versions. It falls back to PyAV and works fine.

## Acknowledgments

DitecT Lab, Columbia University: Prof. Xuan (Sharon) Di, PhD mentor Qi Gao.
Built on [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base) by Hugging Face.
