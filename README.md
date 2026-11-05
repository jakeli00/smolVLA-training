# smolVLA-training

Post-training [SmolVLA](https://huggingface.co/lerobot/smolvla_base) into a **lane-following policy** for the DitecT Lab scaled autonomous car (E1).

The pipeline takes the car's raw driving recordings (ROS 2 bags + front camera video), converts them into a [LeRobot](https://github.com/huggingface/lerobot) dataset, and fine-tunes SmolVLA (450M) to map camera frames to steering and speed. The trained policy is meant to run on the car's Jetson.

## Quick start (training)

The converted dataset is already on Hugging Face, so you can go straight to training. You don't need the raw data or `convert_runs.py`.

1. **Get access.** Send Jake your GitHub and Hugging Face usernames, then accept both invites:
   - GitHub: this repo (`jakeli00/smolVLA-training`)
   - Hugging Face: the [`ditect-lab`](https://huggingface.co/ditect-lab) org, which holds the private datasets
2. **Make a Hugging Face token** with read access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. **Clone, install, and log in:**
   ```bash
   git clone --recurse-submodules https://github.com/jakeli00/smolVLA-training.git
   cd smolVLA-training/lerobot
   uv venv --python 3.12
   uv pip install -e ".[core_scripts,training,smolvla]"
   cd ..
   lerobot/.venv/bin/hf auth login      # paste your token
   ```
4. **Train.** Run the command under [Fine-tune SmolVLA](#2-fine-tune-smolvla). It downloads [`ditect-lab/ditect_aoe_lane`](https://huggingface.co/datasets/ditect-lab/ditect_aoe_lane) (2.8 GB) automatically on the first run.

To check that you can reach the dataset before training:
```bash
lerobot/.venv/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('ditect-lab/ditect_aoe_lane'); print(d.meta.total_episodes, 'episodes,', d.meta.total_frames, 'frames')"
```
It should print `102 episodes, 148371 frames`.

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
└── data/                  # not in git, download from Hugging Face (see "Data")
    ├── raw/               # one folder per run (E1-1022-01 ... E1-1022-40)
    └── lerobot/aoe_lane/  # converted dataset written by convert_runs.py
```

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg (for video encoding and decoding)
- A Hugging Face account with access to the `ditect-lab` org (see [Data](#data))
- A CUDA GPU for training. Apple Silicon (MPS) works for testing but is slow.
- About 10 GB of free disk to rebuild the dataset (4.3 GB raw + 2.8 GB output + temporary frames)

### Tested environment

The dataset on the Hub was built with exactly this setup:

| | Version |
|---|---|
| Machine | MacBook, Apple A18 Pro, macOS 26.6 |
| Python | 3.12.15 (uv 0.12) |
| LeRobot | 0.6.2, submodule commit `3f797de` |
| PyTorch | 2.11.0 |
| rosbags | 0.11.9 |
| PyAV | 15.1.0 |
| OpenCV | 4.13.0 (`opencv-python-headless`, installed by LeRobot) |
| pandas / numpy | 2.3.3 / 2.2.6 |
| huggingface_hub | 1.33.0 |
| ffmpeg | 9.0.2 (Homebrew) |

## Setup

```bash
# 1. Clone with the LeRobot submodule (pinned to the commit above)
git clone --recurse-submodules https://github.com/jakeli00/smolVLA-training.git
cd smolVLA-training

# 2. Create the environment inside lerobot/ and install LeRobot with SmolVLA
cd lerobot
uv venv --python 3.12
uv pip install -e ".[core_scripts,training,smolvla]"
uv pip install rosbags          # ROS 2 bag reader, only needed for convert_runs.py
cd ..

# 3. Log in to Hugging Face (needed to download the private datasets)
lerobot/.venv/bin/hf auth login
```

- If you already cloned without `--recurse-submodules`, run `git submodule update --init`.
- On macOS, install ffmpeg with `brew install ffmpeg`.
- Every command in this README runs from the repo root and uses the venv's Python directly (`lerobot/.venv/bin/...`), so you never need to activate it.

## Data

The data is too large for git, so it lives in two **private** dataset repos in the [ditect-lab](https://huggingface.co/ditect-lab) Hugging Face org:

| Repo | Size | Contents | Needed for |
|---|---|---|---|
| [`ditect-lab/ditect_aoe_lane`](https://huggingface.co/datasets/ditect-lab/ditect_aoe_lane) | 2.8 GB | converted LeRobot dataset | training |
| [`ditect-lab/ditect_aoe_raw`](https://huggingface.co/datasets/ditect-lab/ditect_aoe_raw) | 4.3 GB | raw runs (video + ROS 2 bags) | re-running `convert_runs.py` |

### Getting access

1. Create a free account at [huggingface.co](https://huggingface.co/join) and send your username to Jake. He'll add you to the `ditect-lab` org.
2. Accept the invite at [huggingface.co/ditect-lab](https://huggingface.co/ditect-lab).
3. Create an access token (read access is enough) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and log in on your machine:

```bash
lerobot/.venv/bin/hf auth login
```

### Downloading

For training you don't need to download anything by hand. `lerobot-train --dataset.repo_id=ditect-lab/ditect_aoe_lane` fetches the dataset automatically and caches it in `~/.cache/huggingface/lerobot/`.

To get local copies instead:

```bash
# Converted dataset -> data/lerobot/aoe_lane
lerobot/.venv/bin/hf download ditect-lab/ditect_aoe_lane --repo-type dataset --local-dir data/lerobot/aoe_lane

# Raw runs -> data/raw (only needed to re-run the conversion)
lerobot/.venv/bin/hf download ditect-lab/ditect_aoe_raw --repo-type dataset --local-dir data/raw
```

### Raw run format

```
data/raw/E1-1022-01/
├── video.ts      # front camera, 1280x960 @ 30 fps, mounted upside down
├── frames.csv    # wall-clock time of each video frame
├── meta.json     # car id, start/end time, steering trim, topics
└── bag/          # ROS 2 bag (.db3): /ackermann_cmd and /odom at ~50 Hz
```

Runs 08 and 16 have empty bags and run 12 has no video frames. The script skips all three automatically.

## Usage

### 1. Rebuild the dataset from the raw runs

Skip this if you only want to train, since the converted dataset is already on the Hub. These steps reproduce `ditect-lab/ditect_aoe_lane` from scratch.

**a. Download the raw runs** (4.3 GB) into `data/raw`:

```bash
lerobot/.venv/bin/hf download ditect-lab/ditect_aoe_raw --repo-type dataset --local-dir data/raw
```

You should end up with 40 folders, `data/raw/E1-1022-01` through `data/raw/E1-1022-40`.

**b. Point the script at your copy.** The two paths at the top of `convert_runs.py` are absolute paths on Jake's machine. Change them to your repo location:

```python
raw = Path("/path/to/smolVLA-training/data/raw")
out = Path("/path/to/smolVLA-training/data/lerobot/aoe_lane")
```

**c. Run the conversion:**

```bash
lerobot/.venv/bin/python -u convert_runs.py | tee data/convert_log.txt
```

This takes about 2 hours on the machine above, almost all of it AV1 video encoding. It prints one line per run, for example `E1-1022-01: 4 episodes, 4321 of 5270 frames kept`, and finishes with:

```
Done: 102 episodes, 148371 frames in .../data/lerobot/aoe_lane
```

The script refuses to run if `out` already exists, so delete that folder before rebuilding.

**d. Check the result:**

```bash
lerobot/.venv/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
d=LeRobotDataset('ditect-lab/ditect_aoe_lane', root='data/lerobot/aoe_lane'); \
print(d.meta.total_episodes, d.meta.total_frames); s=d[0]; print(s['task'], s['observation.images.front'].shape, s['action'])"
```

This should print `102 148371`, then `follow the lane torch.Size([3, 480, 640])` and a 2-value action.

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

The episode rules are the constants at the top of the script (`MIN_SPEED`, `MAX_GAP_S`, `MIN_EPISODE_S`). Changing them gives a different episode count.

**e. (Optional) Publish a rebuilt dataset to the org.** This replaces the files on the Hub, so check with Jake first:

```bash
lerobot/.venv/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
LeRobotDataset('ditect-lab/ditect_aoe_lane', root='data/lerobot/aoe_lane').push_to_hub(private=True, license=None)"
```

### 2. Fine-tune SmolVLA

This is a starting command based on the [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla). Tune the batch size and steps for your GPU.

```bash
lerobot/.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=ditect-lab/ditect_aoe_lane \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla_lane \
  --job_name=smolvla_lane \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --trackio.enable=true
```

`--rename_map` maps our camera name onto the camera slot SmolVLA was pretrained with. To train from a local copy, add `--dataset.root=data/lerobot/aoe_lane`. Run `trackio show` to watch the training curves locally.

## Notes

- `/odom` reports speed with the opposite sign of the command, so the script negates it.
- Speed is capped at 0.5 m/s by the car's bridge, and steering trim is -0.12 on every run.
- The data includes intersections and turns, not just straight lane following. A single `"follow the lane"` task may be ambiguous at intersections.
- On first load, LeRobot may print `torchcodec` errors with newer ffmpeg versions. It falls back to PyAV and works fine.

## Acknowledgments

DitecT Lab, Columbia University: Prof. Xuan (Sharon) Di, PhD mentor Qi Gao.
Built on [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base) by Hugging Face.
