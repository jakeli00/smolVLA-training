# smolVLA-training

Lane-following policy training for the DitecT Lab scaled autonomous car (E1).

## convert_runs.py

Converts raw car recordings into a LeRobot v3 dataset.

Each run folder in `data/raw/` holds:
- `video.ts`: front camera, 1280x960 at 30 fps, mounted upside down
- `frames.csv`: wall-clock time of each video frame
- `bag/`: ROS 2 bag with `/ackermann_cmd` (steering, speed) and `/odom` at about 50 Hz

For each run the script:
1. Matches every video frame to the steering and speed command at or just after it, and to the latest odometry reading.
2. Keeps the stretches where the car is moving. Commanded speed must be above 0.05 m/s, stops under 1 s are kept, and stretches under 2 s are dropped. Each stretch becomes one episode.
3. Rotates each frame 180 degrees, resizes it to 640x480 and writes it to the dataset with:
   - `action` = [steering_rad, speed_mps]
   - `observation.state` = [measured speed, previous steering]
   - `task` = "follow the lane"

Output: `data/lerobot/aoe_lane`, repo id `jakeli00/ditect_aoe_lane`. On the AoE 2026 data (40 runs), 37 runs were usable. The result was 102 episodes and 148,371 frames, about 82 minutes of driving.

## Setup

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && uv venv && uv pip install -e ".[smolvla]" && uv pip install rosbags opencv-python
cd .. && lerobot/.venv/bin/python convert_runs.py
```

The script refuses to run if the output folder already exists.
