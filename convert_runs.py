from pathlib import Path
import pandas as pd
import numpy as np
import av
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset


raw = Path("/Users/jakeli/Developer/ditect-vla/data/raw")
out = Path("/Users/jakeli/Developer/ditect-vla/data/lerobot/aoe_lane")

FPS = 30
TASK = "follow the lane"
MIN_SPEED = 0.05      # m/s, below this the car counts as stopped
MAX_GAP_S = 1.0       # stops shorter than this stay inside one episode
MIN_EPISODE_S = 2.0   # moving stretches shorter than this are dropped

# Set up the decoder and teach it the Ackermann message type
typestore = get_typestore(Stores.ROS2_HUMBLE)
typestore.register(get_types_from_msg(
	"float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk",
	"ackermann_msgs/msg/AckermannDrive"))
typestore.register(get_types_from_msg(
	"std_msgs/Header header\nackermann_msgs/AckermannDrive drive",
	"ackermann_msgs/msg/AckermannDriveStamped"))


def moving_segments(frame_speed):
	"""Return (start, end) frame ranges where the car is moving, end exclusive."""
	moving = frame_speed > MIN_SPEED
	edges = np.diff(np.concatenate(([0], moving.astype(int), [0])))
	starts = np.where(edges == 1)[0]
	ends = np.where(edges == -1)[0]

	# Merge stretches separated by short stops
	merged = []
	for s, e in zip(starts, ends):
		if merged and s - merged[-1][1] < MAX_GAP_S * FPS:
			merged[-1][1] = e
		else:
			merged.append([s, e])

	return [(s, e) for s, e in merged if e - s >= MIN_EPISODE_S * FPS]


if out.exists():
	raise SystemExit(f"{out} already exists. Delete it or change `out` before converting again.")

features = {
	"observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
	"observation.state": {"dtype": "float32", "shape": (2,), "names": ["speed_mps", "prev_steering_rad"]},
	"action": {"dtype": "float32", "shape": (2,), "names": ["steering_rad", "speed_mps"]},
}
dataset = LeRobotDataset.create(
	repo_id="jakeli00/ditect_aoe_lane",
	fps=FPS,
	features=features,
	root=out,
	robot_type="ditect_e1",
)


for files in sorted(raw.iterdir()):
	if not files.is_dir():
		continue
	run_id = files.name
	frames_file = files / "frames.csv"
	frames = pd.read_csv(frames_file)
	frames = frames.drop(columns = "pts_90khz").dropna()
	if frames.empty:
		continue
	frame_times = frames["wall_time"].to_numpy()

	# Read the two topics needed from this run's bag
	cmd_t, steer, speed = [], [], []
	odom_t, odom_v = [], []

	with Reader(files / "bag") as reader:
		wanted = [c for c in reader.connections if c.topic in ("/ackermann_cmd", "/odom")]
		for conn, timestamp, rawdata in reader.messages(connections=wanted):
			msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
			if conn.topic == "/ackermann_cmd":
				cmd_t.append(timestamp / 1e9)
				steer.append(msg.drive.steering_angle)
				speed.append(msg.drive.speed)
			else:
				odom_t.append(timestamp / 1e9)
				odom_v.append(-msg.twist.twist.linear.x)

	if len(cmd_t) == 0 or len(odom_t) == 0:
		continue
	cmd_t = np.array(cmd_t)
	steer = np.array(steer)
	speed = np.array(speed)
	odom_t = np.array(odom_t)
	odom_v = np.array(odom_v)
	cmd_idx = np.searchsorted(cmd_t, frame_times, side = "left")
	cmd_idx = np.clip(cmd_idx, 0, len(cmd_t) - 1)
	frame_steer = steer[cmd_idx]
	frame_speed = speed[cmd_idx]
	
	odom_idx = np.searchsorted(odom_t, frame_times, side = "right") - 1
	odom_idx = np.clip(odom_idx, 0, len(odom_t) - 1)
	frame_odom_v = odom_v[odom_idx]
	frame_prev_steer = np.concatenate(([frame_steer[0]], frame_steer[:-1]))

	# Which episode each frame belongs to, -1 means skip (car stopped)
	segments = moving_segments(frame_speed)
	if not segments:
		continue
	episode_of = np.full(len(frame_times), -1)
	for n, (s, e) in enumerate(segments):
		episode_of[s:e] = n
	print(f"{run_id}: {len(segments)} episodes, {int((episode_of >= 0).sum())} of {len(frame_times)} frames kept")

	current = -1
	container = av.open(str(files / "video.ts"))
	for i, frame in enumerate(container.decode(video=0)):
		if i >= len(frame_times):
			break

		# Leaving an episode: save it
		if episode_of[i] != current and current >= 0:
			dataset.save_episode()
		current = episode_of[i]
		if current < 0:
			continue

		img = frame.to_ndarray(format="rgb24")
		img = np.rot90(img, 2)
		img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
		action = np.array([frame_steer[i], frame_speed[i]], dtype=np.float32)
		state = np.array([frame_odom_v[i], frame_prev_steer[i]], dtype=np.float32)

		dataset.add_frame({
			"observation.images.front": img,
			"observation.state": state,
			"action": action,
			"task": TASK,
		})
	container.close()

	# Run ended while still inside an episode
	if current >= 0:
		dataset.save_episode()

dataset.finalize()
print(f"Done: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames in {out}")
