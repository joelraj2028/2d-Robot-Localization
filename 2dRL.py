import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from collections import deque
from scipy.ndimage import distance_transform_edt
from PIL import Image
import math
import random
import os

#
#
# |PARAMETERS|

# Simulation
maze_image = "maze1.png"
num_targets = 4
min_dist = 2

# raycasting
fov_deg = 45
fov = np.deg2rad(fov_deg)
num_fov_rays = 30
ray_max_range = 30
ray_noise_std = 0
ray_step = 0.5
plot_every_nth_ray = 1

# HMM
alpha = 0.5  # sensor noise parameter

# Animation
fps = 8
interval_ms = int(1000 / fps)

# Randomness
random_seed = None
shuffle_bfs_neighbors = True
perturb_prob = 0.12
max_perturb_attempts = 3
#
#
#

# Progress Bar
PB_WIDTH = 50
def print_progress(label, current, total, width=PB_WIDTH):
    if total <= 0:
        frac = 1.0
    else:
        frac = float(current) / float(total)
        frac = max(0.0, min(1.0, frac))
    pct = int(round(frac * 100))
    blocks = int(round((pct / 100.0) * width))
    bar = "█" * blocks + "." * (width - blocks)
    print(f"\r{label}: |{bar}| {pct}%", end='', flush=True)
    if current >= total:
        print()  # newline when complete


# Environment
class Environment:
    def __init__(self, map_file):
        self.map = np.array(Image.open(map_file).convert("L")) / 255.0
        self.free = self.map > 0.5
        self.X = np.array(np.where(self.free)).T
        self.num_states = self.X.shape[0]
        self.state_index = -np.ones(self.map.shape, dtype=int)
        for i, (r, c) in enumerate(self.X):
            self.state_index[r, c] = i
        self.neighbors = []
        rows, cols = self.map.shape
        for (r, c) in self.X:
            neigh = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < rows and 0 <= cc < cols and self.free[rr, cc]:
                    idx = self.state_index[rr, cc]
                    if idx >= 0:
                        neigh.append(idx)
            self.neighbors.append(neigh)

    def raycast_fov(self, pos, facing_angle, fov=0.0, num_rays=1, max_range=None, step=0.2, noise_std=0.0):
        r0, c0 = float(pos[0]), float(pos[1])
        rows, cols = self.map.shape
        if num_rays == 1 or fov == 0:
            angles = np.array([facing_angle])
        else:
            angles = np.linspace(facing_angle - fov/2, facing_angle + fov/2, num_rays)
        distances = np.zeros(len(angles), dtype=float)
        for i, ang in enumerate(angles):
            dr = np.sin(ang); dc = np.cos(ang)
            rr, cc = r0, c0; dist = 0.0
            while True:
                rr += dr * step; cc += dc * step; dist += step
                if not (0 <= rr < rows and 0 <= cc < cols):
                    distances[i] = min(dist, max_range) if max_range else dist; break
                if not self.free[int(rr), int(cc)]:
                    distances[i] = dist; break
                if max_range and dist >= max_range:
                    distances[i] = max_range; break
                if dist > max(rows, cols) * 2:
                    distances[i] = dist; break
            if noise_std > 0:
                distances[i] += np.random.normal(0, noise_std)
        return np.clip(distances, 0.0, max_range if max_range else np.inf)


# Random seed
if random_seed is not None:
    np.random.seed(random_seed)
    random.seed(random_seed)
else:
    random.seed()

# Load environment
if not os.path.exists(maze_image):
    raise FileNotFoundError(f"Maze image '{maze_image}' not found.")
env = Environment(maze_image)
rows, cols = env.map.shape

# Distance and Heat Map
from scipy.ndimage import distance_transform_edt
distance_to_wall = distance_transform_edt(env.map > 0).astype(float)

def belief_to_heatmap(belief, env, distance_to_wall):
    """Convert belief vector into a 2D heatmap aligned with the maze image."""
    heatmap = np.zeros_like(env.map, dtype=float)

    for i, (r, c) in enumerate(env.X):
        heatmap[int(r), int(c)] = belief[i]

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = np.flipud(heatmap)
    return heatmap

#
#
# |ROBOT PATH|

# Pick a starting point
valid_starts = np.array(env.X)
start_r, start_c = valid_starts[np.random.randint(len(valid_starts))]
start_idx = env.state_index[start_r, start_c]

# Pick random targets
central_mask = distance_to_wall >= min_dist
central_indices = [i for i, (r,c) in enumerate(env.X) if central_mask[int(r), int(c)]]
central_indices = [i for i in central_indices if i != start_idx]
target_indices = np.random.choice(central_indices, size=num_targets, replace=False)
points = [start_idx] + list(target_indices)

# Create Path
def bfs_path_randomized(start_idx, goal_idx, neighbors, allowed_set):
    queue = deque([start_idx])
    visited = {start_idx: None}
    while queue:
        current = queue.popleft()
        if current == goal_idx:
            path=[]
            while current is not None:
                path.append(current)
                current = visited[current]
            return path[::-1]
        neighs = list(neighbors[current])
        if shuffle_bfs_neighbors:
            random.shuffle(neighs)
        for n in neighs:
            if n not in visited and n in allowed_set:
                visited[n] = current
                queue.append(n)
    return []

def perturb_path(idx_path):
    out = list(idx_path)
    N = len(out)
    for i in range(1,N-1):
        if random.random() < perturb_prob:
            base = out[i]
            for _ in range(max_perturb_attempts):
                neighs = env.neighbors[base]
                if not neighs: break
                alt = random.choice(neighs)
                if alt != out[i-1] and alt != out[i+1]:
                    out[i] = alt
                    break
    return out
    
# Build trajectory
trajectory_indices = []
allowed_set = set(central_indices) if len(central_indices)>0 else set(range(env.num_states))
for i in range(len(points)-1):
    seg = bfs_path_randomized(points[i], points[i+1], env.neighbors, allowed_set)
    if not seg:
        seg = bfs_path_randomized(points[i], points[i+1], env.neighbors, set(range(env.num_states)))
        if not seg:
            raise RuntimeError("No path found between points.")
    seg = perturb_path(seg)
    trajectory_indices.extend(seg[:-1])
trajectory_indices.append(points[-1])
trajectory_state_indices = np.array(trajectory_indices)
trajectory = np.array([env.X[idx] for idx in trajectory_indices], dtype=float)
T = len(trajectory)
#
#
#

# Measurements
def get_facing_angle(current, next_pos):
    dr = next_pos[0]-current[0]; dc = next_pos[1]-current[1]
    return math.atan2(dr, dc)

all_ranges = np.zeros((T, num_fov_rays), dtype=float)
all_facings = np.zeros(T, dtype=float)
for frame in range(T):
    if frame < T-1:
        facing = get_facing_angle(trajectory[frame], trajectory[frame+1])
    else:
        facing = get_facing_angle(trajectory[frame-1], trajectory[frame])
    all_facings[frame] = facing
    all_ranges[frame] = env.raycast_fov(
        pos=trajectory[frame], facing_angle=facing, fov=fov,
        num_rays=num_fov_rays, max_range=ray_max_range,
        step=ray_step, noise_std=ray_noise_std
    )

# Scans
state_scans = []
for idx in range(env.num_states):
    pos = env.X[idx]
    state_scans.append(env.raycast_fov(
        pos, facing_angle=0.0, fov=fov, num_rays=num_fov_rays,
        max_range=ray_max_range, step=ray_step, noise_std=0.0
    ))

#
#
# --|HMM|--

# Parameters
spread_steps = 3          # how many graph steps probability can spread each timestep
facing_bins = 36          # discretize orientation for caching (36 -> 10° bins)
cache_enabled = True
log_eps = 1e-300


_expected_scan_cache = {}

def facing_to_bin(angle):
    a = angle % (2*np.pi)
    b = int(np.round(a / (2*np.pi) * facing_bins)) % facing_bins
    return b

def get_expected_scan_for_state_and_facing(state_idx, facing):
    """
    Return expected scan (num_fov_rays,) for a given state index and facing angle.
    Uses a small cache for (state_idx, facing_bin).
    """
    if not cache_enabled:
        return env.raycast_fov(pos=env.X[state_idx], facing_angle=facing,
                                fov=fov, num_rays=num_fov_rays, max_range=ray_max_range,
                                step=ray_step, noise_std=0.0)
    b = facing_to_bin(facing)
    key = (int(state_idx), int(b))
    if key in _expected_scan_cache:
        return _expected_scan_cache[key]
    
    bin_angle = (b / facing_bins) * 2*np.pi
    scan = env.raycast_fov(pos=env.X[state_idx], facing_angle=bin_angle,
                            fov=fov, num_rays=num_fov_rays, max_range=ray_max_range,
                            step=ray_step, noise_std=0.0)
    _expected_scan_cache[key] = scan
    return scan

# Baysian Filtering
belief = np.zeros(env.num_states)
belief[start_idx] = 1.0
beliefs_over_time = []
predicted_positions = []

neighbors_list = env.neighbors

T = len(trajectory)
K = env.num_states

print_progress("Filtering", 0, T)
for t in range(T):
    meas = all_ranges[t].copy()
    facing_t = all_facings[t]

    belief_pred = np.zeros_like(belief)

    nz_idx = np.nonzero(belief > 1e-12)[0]
    for n in nz_idx:
        p_n = belief[n]
        fringe = {n}
        frontier = {n}
        for _ in range(spread_steps):
            new_frontier = set()
            for s in frontier:
                for nb in neighbors_list[s]:
                    if nb not in fringe:
                        new_frontier.add(nb)
            fringe.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        if len(fringe) == 0:
            continue
        share = p_n / float(len(fringe))
        for s in fringe:
            belief_pred[s] += share

    # small diffusion to avoid zeroing everything
    diffusion = 1e-4
    belief_pred = (1.0 - diffusion) * belief_pred + diffusion / float(K)

    # mearment update
    log_bel = np.log(belief_pred + log_eps)
    for s in range(K):
        expected = get_expected_scan_for_state_and_facing(s, facing_t)
        gamma = 0.1
        sig = alpha * np.maximum(expected, gamma)
        dif = (meas - expected) / (sig + 1e-12)
        ll = -0.5 * np.sum(dif * dif) - np.sum(np.log(sig + 1e-12)) - (len(meas) * 0.5 * np.log(2*np.pi))
        log_bel[s] += ll

    # normalize w/ log-sum
    max_log = np.max(log_bel)
    belief = np.exp(log_bel - max_log)
    ssum = belief.sum()
    if ssum <= 0:
        # fallback(shouldn't happen)
        belief = np.ones_like(belief) / float(K)
    else:
        belief /= ssum

    beliefs_over_time.append(belief.copy())

    # predicted position
    pred_r = np.sum(env.X[:,0] * belief)
    pred_c = np.sum(env.X[:,1] * belief)
    predicted_positions.append([pred_r, pred_c])

    print_progress("Filtering", t+1, T)

predicted_positions = np.array(predicted_positions)

# Offline virterbi
print_progress("Computing L matrix", 0, T)
L = np.full((T, K), -np.inf)
for t in range(T):
    meas = all_ranges[t]
    facing_t = all_facings[t]
    for s in range(K):
        expected = get_expected_scan_for_state_and_facing(s, facing_t)
        sig = alpha * np.maximum(expected, 0.1)
        dif = (meas - expected) / (sig + 1e-12)
        ll = -0.5 * np.sum(dif * dif) - np.sum(np.log(sig + 1e-12)) - (len(meas) * 0.5 * np.log(2*np.pi))
        L[t, s] = ll
    print_progress("Computing L matrix", t+1, T)

# Viterbi DP
V = np.full((T, K), -np.inf)
B = np.zeros((T, K), dtype=int)

# initial
V[0, :] = L[0, :] + np.log(1.0 / float(K))

print_progress("Viterbi DP", 0, T)
for t in range(1, T):
    for s in range(K):
        best_val = -np.inf
        best_prev = 0
        candidates = neighbors_list[s] + [s]
        for pstate in candidates:
            denom = max(1, len(neighbors_list[pstate]) + 1)
            trans_log = -np.log(denom)
            val = V[t-1, pstate] + trans_log
            if val > best_val:
                best_val = val
                best_prev = pstate
        V[t, s] = best_val + L[t, s]
        B[t, s] = best_prev
    print_progress("Viterbi DP", t+1, T)

# backtrace
state = int(np.argmax(V[-1, :]))
path_states = [state]
for t in range(T-1, 0, -1):
    state = int(B[t, state])
    path_states.append(state)
path_states = list(reversed(path_states))
viterbi_path = np.array(path_states, dtype=int)
viterbi_positions = env.X[viterbi_path]
#
#
#

#
#
# |Animation|

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(env.map, cmap="gray", origin="upper")
ax.set_xlim(-0.5, cols - 0.5)
ax.set_ylim(rows - 0.5, -0.5)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_yticks([])

robot_dot, = ax.plot([], [], 'ro', markersize=6)
path_line, = ax.plot([], [], 'y--', linewidth=1.5)
pred_dot, = ax.plot([], [], 'go', markersize=6)
pred_line, = ax.plot([], [], 'g-', linewidth=1.5)

# Heatmap overlay
initial_heatmap = belief_to_heatmap(beliefs_over_time[0], env, distance_to_wall)
initial_heatmap = np.flipud(initial_heatmap)   # 🔥 IMPORTANT FIX
heatmap_im = ax.imshow(initial_heatmap, cmap="hot", alpha=0.45, origin="upper")

# Error display text
error_text = ax.text(
    0 * cols,       # positions
    0 * rows,
    "",
    fontsize=10,
    ha='left',
    va='top',
    color='black',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)


def init():
    robot_dot.set_data([], [])
    path_line.set_data([], [])
    pred_dot.set_data([], [])
    pred_line.set_data([], [])
    heatmap_im.set_data(initial_heatmap)
    error_text.set_text("")
    return [robot_dot, path_line, pred_dot, pred_line, heatmap_im, error_text]

def update_main(frame):
    r0, c0 = trajectory[frame]
    robot_dot.set_data([c0], [r0])

    path_line.set_data(
        trajectory[:frame+1, 1],
        trajectory[:frame+1, 0]
    )

    est_r, est_c = predicted_positions[frame]
    pred_dot.set_data([est_c], [est_r])

    pred_line.set_data(
        predicted_positions[:frame+1, 1],
        predicted_positions[:frame+1, 0]
    )

    heatmap = belief_to_heatmap(beliefs_over_time[frame], env, distance_to_wall)
    heatmap = np.flipud(heatmap)
    heatmap_im.set_data(heatmap)

    # error calc
    true_r, true_c = trajectory[frame]
    instant_error = np.sqrt((true_r - est_r)**2 + (true_c - est_c)**2)

    avg_error = np.mean([
        np.sqrt((trajectory[i,0] - predicted_positions[i,0])**2 +
                (trajectory[i,1] - predicted_positions[i,1])**2)
        for i in range(frame + 1)
    ])

    error_text.set_text(
        f"Error: {instant_error:.2f}\nAvg error: {avg_error:.2f}"
    )

    return [robot_dot, path_line, pred_dot, pred_line, heatmap_im, error_text]



ani = animation.FuncAnimation(
    fig, update_main, frames=T,
    init_func=init, interval=interval_ms, blit=True
)

out_name = "Simulation.gif"
ani.save(out_name, writer="pillow", fps=fps)
plt.close(fig)
print(f"Saved {out_name}")

# Debug GIF
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.imshow(env.map, cmap="gray", origin="upper")
ax2.set_xlim(-0.5, cols - 0.5)
ax2.set_ylim(rows - 0.5, -0.5)
ax2.invert_yaxis()
ax2.set_xticks([])
ax2.set_yticks([])

robot_dot2, = ax2.plot([], [], 'ro', markersize=6)
path_line2, = ax2.plot([], [], 'y--', linewidth=1.5)
lc2 = LineCollection([], linewidths=0.8)
ax2.add_collection(lc2)
end_scatter2 = ax2.scatter([], [], s=18, marker='o')

def init2():
    robot_dot2.set_data([], [])
    path_line2.set_data([], [])
    lc2.set_segments([])
    end_scatter2.set_offsets(np.empty((0, 2)))
    return [robot_dot2, path_line2, lc2, end_scatter2]

def update_debug(frame):
    r0, c0 = trajectory[frame]
    facing = all_facings[frame]
    measurement = all_ranges[frame]

    robot_dot2.set_data([c0], [r0])
    path_line2.set_data(
        trajectory[:frame+1, 1],
        trajectory[:frame+1, 0]
    )

    angles = np.linspace(
        facing - fov/2,
        facing + fov/2,
        num_fov_rays
    )

    segments = []
    endpoints = []

    for i, (ang, rng) in enumerate(zip(angles, measurement)):
        if i % plot_every_nth_ray != 0:
            continue

        dr = np.sin(ang)
        dc = np.cos(ang)

        r_end = r0 + dr * rng
        c_end = c0 + dc * rng

        segments.append([(c0, r0), (c_end, r_end)])
        endpoints.append([c_end, r_end])

    lc2.set_segments(segments)
    end_scatter2.set_offsets(np.array(endpoints) if endpoints else np.empty((0, 2)))

    return [robot_dot2, path_line2, lc2, end_scatter2]


ani2 = animation.FuncAnimation(
    fig2, update_debug, frames=T,
    init_func=init2, interval=interval_ms, blit=True
)

out_name2 = "Debug.gif"
ani2.save(out_name2, writer="pillow", fps=fps)
plt.close(fig2)
print(f"Saved {out_name2}")

print("Done!")
#
#
#
