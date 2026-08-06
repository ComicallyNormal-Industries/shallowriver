import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os

# Bone connections for the 34-keypoint model
BONES = [
    (6, 15), (20, 21), (6, 3), (3, 0), (20, 1), (21, 2),
    (0, 1), (0, 2), (1, 4), (2, 5), (20, 22), (22, 24),
    (21, 23), (23, 25)
]

def load_data(filepath):
    """Parses labeled 'Keypoint_X: x, y, z' files separated by '--- Frame Start ---'."""
    frames = []
    current_frame = []

    try:
        f = open(filepath, 'r')
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Run the pipeline first to generate 3D output.")
        return frames

    with f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if "--- Frame Start ---" in line:
                if current_frame:
                    # VALIDATION: Only accept frames that have exactly 34 keypoints
                    if len(current_frame) == 34:
                        frames.append(np.array(current_frame, dtype=np.float32))
                    else:
                        print(f"Warning: Skipping incomplete frame ending near line {line_num} (Found {len(current_frame)}/34 keypoints)")
                current_frame = []

            elif line.startswith("Keypoint_"):
                # A line can arrive here truncated (e.g. "Keypoint_15" with no ": x, y, z"
                # yet) if the C++ pipeline is still actively writing this file while it's
                # being read -- treat that the same as any other malformed line: warn and
                # skip, rather than crashing on a missing ':'.
                if ':' not in line:
                    print(f"Warning: Malformed line (missing ':') on line {line_num}: '{line}'")
                    continue
                try:
                    # Extract coordinates after the colon ':'
                    coords_str = line.split(':')[1]
                    coords = [float(val.strip()) for val in coords_str.split(',')]

                    # VALIDATION: Ensure we have exactly X, Y, and Z
                    if len(coords) == 3:
                        current_frame.append(coords)
                    else:
                        print(f"Warning: Malformed coordinates on line {line_num}: '{line}'")
                except ValueError:
                    print(f"Warning: Could not parse numbers on line {line_num}: '{line}'")

        # Append the final frame if file doesn't end with a delimiter
        if current_frame:
            if len(current_frame) == 34:
                frames.append(np.array(current_frame, dtype=np.float32))
            else:
                print(f"Warning: Skipping incomplete final frame. (Found {len(current_frame)}/34 keypoints)")

    return frames

def draw_skeleton(ax, joints, title):
    ax.clear()

    # Expanded limits to capture millimeter scale variations across frames
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 4000)
    ax.set_zlim(-1000, 4000)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)

    # Plot joints (34 keypoints)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=25)

    # Plot bones
    for start, end in BONES:
        if start < len(joints) and end < len(joints):
            xs = [joints[start, 0], joints[end, 0]]
            ys = [joints[start, 1], joints[end, 1]]
            zs = [joints[start, 2], joints[end, 2]]
            ax.plot(xs, ys, zs, c='b', linewidth=2)

def update_plot(val):
    frame_idx = int(slider.val)

    draw_skeleton(ax1, all_frames_1[frame_idx], f"Camera 1 - Frame: {frame_idx} / {max_frame}")
    draw_skeleton(ax2, all_frames_2[frame_idx], f"Camera 2 - Frame: {frame_idx} / {max_frame}")

    fig.canvas.draw_idle()

# --- Main Setup ---
# Using os.path to ensure it finds the files regardless of where you run the script from
script_dir = os.path.dirname(os.path.abspath(__file__))
filepath_1 = os.path.join(script_dir, '..', 'build', 'output', '3d_key_points_1.txt')
filepath_2 = os.path.join(script_dir, '..', 'build', 'output', '3d_key_points_2.txt')

all_frames_1 = load_data(filepath_1)
all_frames_2 = load_data(filepath_2)

if not all_frames_1:
    print(f"No frames found! Check if '{filepath_1}' exists and contains data.")
    exit()

if not all_frames_2:
    print(f"No frames found! Check if '{filepath_2}' exists and contains data.")
    exit()

print(f"Successfully loaded {len(all_frames_1)} valid frames for camera 1.")
print(f"Successfully loaded {len(all_frames_2)} valid frames for camera 2.")

# The two cameras run independently and won't have the same frame count -- scrub
# both in lockstep over the range they both actually have data for.
max_frame = min(len(all_frames_1), len(all_frames_2)) - 1
if max_frame < 0:
    print("One of the files has no valid frames.")
    exit()

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
plt.subplots_adjust(bottom=0.25)

slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(slider_ax, 'Frame', 0, max_frame, valinit=0, valstep=1)
slider.on_changed(update_plot)

update_plot(0)
plt.show()
