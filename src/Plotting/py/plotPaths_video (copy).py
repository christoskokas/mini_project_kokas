import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

file_paths = ["ground_truth_traj_exp_07.txt", "dual_cam_traj_exp_07.txt", "orbslam3_exp_07.txt"]
colors = ['b', 'r', 'g','y']
connect_points = True

fig, ax = plt.subplots()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Experiment Trajectories")
ax.grid(True)

scatter_plots = []
for i, file_path in enumerate(file_paths):
    if i == 0 :
        scatter_plots.append(ax.scatter([], [], c=colors[i], label="Ground Truth", s=2))
    if i == 1 :
        scatter_plots.append(ax.scatter([], [], c=colors[i], label="Dual Cam", s=2))
    if i == 2 :
        scatter_plots.append(ax.scatter([], [], c=colors[i], label="ORB-SLAM3", s=2))

def update(frame):
    artists = []
    file_ended = False
    prev_points = []
    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            lines = f.readlines()
            if frame >= len(lines):
                file_ended = True
                continue

            values = lines[frame].split()
            if i==0:
                x = -float(values[7])
                y = float(values[11])
            else:
                x = float(values[3])
                y = float(values[11])

            if prev_points and connect_points and  i < len(prev_points):
                ax.plot([prev_points[i][0], x], [prev_points[i][1], y], colors[i], linewidth=1)

            scatter_plots[i].set_offsets(np.c_[np.append(scatter_plots[i].get_offsets()[:,0], x), 
                                                np.append(scatter_plots[i].get_offsets()[:,1], y)])
            artists.append(scatter_plots[i])

            prev_points.append((x, y))

        if i == 0 and frame == 0:
            ax.plot(x, y, marker="o", markersize = 10, markerfacecolor="green",markeredgecolor='none', label='start', linestyle='None')
            plt.plot([], [], ' ', label="Featureless\n Object", marker='s', markersize=10, markerfacecolor='none', markeredgecolor='b', mew=2)

    if file_ended and frame == num_frames-1:
        ani.event_source.stop()

    return artists

# Set up the animation
num_frames = max([len(open(file_path).readlines()) for file_path in file_paths])
ani = FuncAnimation(fig, update, frames=num_frames, interval=66, blit=True)
plt.ylim(-0.2, 2)
plt.xlim(-0.2, 1)
plt.legend()
plt.show()