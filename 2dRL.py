import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import threading


# Parameters
grid_size = (20, 20)
measure_rate = 0.1        # seconds between measurements
dt = 0.1                 # timestep for simulation updates
motion_noise = 0.01
measurement_noise = 1.0   # noise added to measured point
memory_length = 5         # number of past measurements to consider


# Initialize belief and models

belief = np.zeros(grid_size)
belief[0, 0] = 1.0         # start with full certainty at (0,0)
true_pos = np.array([0.0, 0.0])
measured_pos = np.array([0.0, 0.0])
past_measurements = [measured_pos.copy()]
running = True
pending_move = None
lock = threading.Lock()


# Helper functions

def clamp_pos(pos):
    pos[0] = np.clip(pos[0], 0, grid_size[0]-1)
    pos[1] = np.clip(pos[1], 0, grid_size[1]-1)
    return pos

def move(belief):
    """Predict step: diffuse belief to neighboring cells."""
    new_belief = np.zeros_like(belief)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            neighbors = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
            for di,dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
                    new_belief[ni, nj] += (1 - motion_noise) * belief[i,j] / len(neighbors)
                    
            # Random diffusion
            new_belief[i,j] += motion_noise * belief[i,j] / np.prod(grid_size)
    new_belief /= np.sum(new_belief)
    return new_belief

def sense(belief, meas_pos):
    """Update belief based on a noisy measurement."""
    likelihood = np.zeros_like(belief)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            dist = np.sqrt((i - meas_pos[0])**2 + (j - meas_pos[1])**2)
            likelihood[i,j] = np.exp(-dist**2 / (2 * measurement_noise**2))
    updated_belief = likelihood * belief
    updated_belief /= np.sum(updated_belief)
    return updated_belief


# control

def on_key(event):
    global pending_move, running
    if event.key == 'up':    pending_move = np.array([1, 0])   # move up increases row index
    elif event.key == 'down': pending_move = np.array([-1, 0])
    elif event.key == 'left': pending_move = np.array([0, -1])
    elif event.key == 'right': pending_move = np.array([0, 1])
    elif event.key == 'q':
        running = False
        plt.close()
        

# Real-time update loop

def realtime_loop():
    global belief, true_pos, measured_pos, past_measurements, pending_move
    t_last_measure = time.time()
    while running:
        with lock:
            # Apply user movement
            if pending_move is not None:
                true_pos[:] = clamp_pos(true_pos + pending_move)
                pending_move = None

            # Predict step
            belief[:] = move(belief)

            # Measurement update periodically
            if time.time() - t_last_measure >= measure_rate:
                noisy_meas = true_pos + np.random.normal(0, measurement_noise, size=2)
                noisy_meas = clamp_pos(noisy_meas)
                past_measurements.append(noisy_meas)
                if len(past_measurements) > memory_length:
                    past_measurements.pop(0)
                
                # Apply sense for each measurement in memory
                belief_copy = belief.copy()
                for m in past_measurements:
                    belief_copy = sense(belief_copy, m)
                belief[:] = belief_copy

                measured_pos[:] = noisy_meas
                t_last_measure = time.time()

        time.sleep(dt)


# Visualization setup

plt.ion()
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(belief, cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(belief))
true_dot, = ax.plot([], [], 'bo', markersize=6, label='True Position')
meas_dot, = ax.plot([], [], 'go', markersize=6, label='Measured')
ax.set_title("Robot Localization Problem")
ax.set_xlim(0, grid_size[1])
ax.set_ylim(0, grid_size[0])
ax.legend(loc='upper right')
fig.canvas.mpl_connect('key_press_event', on_key)


# Start real-time thread

thread = threading.Thread(target=realtime_loop, daemon=True)
thread.start()
print("Use arrow keys to move (one step per press). 'q' to quit.")


# Main visualization loop

try:
    while running:
        with lock:
            true_dot.set_data([true_pos[1]], [true_pos[0]])
            meas_dot.set_data([measured_pos[1]], [measured_pos[0]])
            im.set_data(belief)
            im.set_clim(vmin=0, vmax=np.max(belief))
        plt.pause(0.001)
except KeyboardInterrupt:
    running = False
finally:
    plt.close()
