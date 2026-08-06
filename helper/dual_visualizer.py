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

# NVIDIA BodyPose3DNet 34-keypoint layout (indices used for auto-orientation).
# Derived from the bone-index tables in NVIDIA's deepstream-bodypose-3d reference app.
PELVIS = 0
LEFT_HIP = 1
RIGHT_HIP = 2
NECK = 6
LEFT_SHOULDER = 20
RIGHT_SHOULDER = 21

def compute_orientation(joints):
    """
    Builds an orthonormal (right, forward, up) basis anchored at the pelvis, so every
    frame can be rotated into a consistent "feet down, facing the viewer" pose
    regardless of how the person happened to be oriented relative to the camera.

    up:      pelvis -> neck (spine axis). Aligning this with +Z puts the head above
             the pelvis and the feet/ankles below it -- feet down.
    lateral: shoulder line (right_shoulder - left_shoulder), made orthogonal to `up`.
             Falls back to the hip line if the shoulders are degenerate (e.g. a
             low-confidence frame where the model returned near-identical/zero points).
    forward: cross(up, lateral). shallowriver's pose3d is a standard pinhole
             back-projection (X right, Y down, Z depth away from the camera), so a
             person facing the camera has right_shoulder.x < left_shoulder.x (their
             anatomical right appears on the image's left, same as facing a mirror).
             Working through cross(up, lateral) with that sign convention lands
             `forward` pointing along -Z (i.e. toward the camera) whenever the person
             actually faces it -- so aligning `forward` with the plot's +Y, combined
             with the view angle set in draw_skeleton(), puts their front toward the
             viewer.

    Returns (rotation_matrix, pelvis) so the caller can do:
        local_joints = (world_joints - pelvis) @ rotation_matrix.T
    or None if the pose is too degenerate (e.g. all-zero/duplicate points) to derive
    a stable basis from, in which case the frame is left unrotated.
    """
    pelvis = joints[PELVIS]
    up = joints[NECK] - pelvis
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        return None
    up = up / up_norm

    def orthogonalized_lateral(a, b):
        raw = joints[b] - joints[a]
        raw = raw - np.dot(raw, up) * up
        n = np.linalg.norm(raw)
        return raw / n if n > 1e-6 else None

    lateral = orthogonalized_lateral(LEFT_SHOULDER, RIGHT_SHOULDER)
    if lateral is None:
        lateral = orthogonalized_lateral(LEFT_HIP, RIGHT_HIP)
    if lateral is None:
        return None

    forward = np.cross(up, lateral)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return None
    forward = forward / forward_norm

    # Re-derive lateral from forward/up so the basis is exactly orthonormal even
    # after the fallback/rounding above.
    lateral = np.cross(forward, up)

    # Rows are the new basis vectors expressed in world coordinates -- this is the
    # rotation matrix that maps world-space points into the (lateral, forward, up) frame.
    rotation_matrix = np.stack([lateral, forward, up], axis=0)
    return rotation_matrix, pelvis

def align_skeleton(joints):
    """Returns a copy of joints re-oriented to a consistent feet-down,
    facing-the-viewer pose. Falls back to the original (unrotated) joints if the
    frame is too degenerate to compute a stable orientation from."""
    result = compute_orientation(joints)
    if result is None:
        return joints
    rotation_matrix, pelvis = result
    return (joints - pelvis) @ rotation_matrix.T

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

    joints = align_skeleton(joints)

    # Pelvis-centered limits: comfortably fits a full standing body (head/arms above
    # and to the sides, feet below) now that every frame is re-oriented the same way.
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-1200, 1200)
    ax.set_zlim(-1200, 1200)

    # elev/azim chosen so +Z renders up on screen and +Y (the "forward" axis computed
    # in compute_orientation) renders toward the viewer -- verified empirically via
    # mplot3d's proj_transform, not just assumed from the azim convention.
    ax.view_init(elev=15, azim=90)

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
