import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# List of up to 4 file paths to text files
# file_paths = ["ground_truth.txt","single_cam.txt", "dual_cam.txt", "ORB_SLAM3.txt"]
home_path = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/build/devel/lib/vio_slam"
file_paths = [home_path + "/ground_truth_traj.txt"]
# file_paths = [home_path + "/ground_truth_traj.txt", home_path + "/single_cam_traj.txt", home_path + "/DcamTrajectory.txt", "ORB_SLAM3.txt"]

colors = ['b', 'y', 'g','r']

connect_points = True

# Initialize lists to store x, y, z values from all files
all_xs, all_ys, all_zs = [], [], []


# Loop over the file paths
for i, file_path in enumerate(file_paths):
    # Open the file and read the lines
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract the x, y, z values from each line and store them in lists
    xs, ys, zs = [], [], []
    for line in lines:
        values = line.split()
        xs.append(float(values[3]))
        ys.append(float(values[7]))
        zs.append(float(values[11]))

    # Store the x, y, z values in the overall lists
    all_xs.extend(xs)
    all_ys.extend(ys)
    all_zs.extend(zs)

    # Get the file name without the .txt extension
    file_name = file_path.split(".")[0]
    filename = os.path.basename(file_path)
    base_name, extension = os.path.splitext(filename)


    # Plot the x, y, z values as a scatter plot with optional lines connecting the points
    if i == 0:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(xs[0], ys[0], zs[0], color='g', s=100, label='start')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='k', s=100, label='end')
    if connect_points:
        ax.plot(xs, ys, zs, label=base_name)
    else:
        ax.scatter(xs, ys, zs, s=1, label=base_name, color = colors[i])

        
    # Add a sphere symbol of green color at the start of the trajectory and a sphere symbol of red color at the end

    if i == 2:
        for j in range(30, len(xs), 15):
            if j+15 < len(xs):
                dx = xs[j+1] - xs[j]
                dy = ys[j+1] - ys[j]
                dz = zs[j+1] - zs[j]
                ax.quiver(xs[j], ys[j], zs[j], dx, dy, dz, length=0.01, arrow_length_ratio=10, normalize=True, color=colors[i])

    # Add arrows on each trajectory that show the heading of the trajectory
    # if i != 0:
    # xdir = np.diff(xs)
    # ydir = np.diff(ys)
    # zdir = np.diff(zs)
    # ax.quiver(xs[:-1], ys[:-1], zs[:-1], xdir, ydir, zdir, color = colors[i], arrow_length_ratio=0.1)

# Calculate the range and mean of all x, y, z values
ranges = [
    max(all_xs) - min(all_xs),
    max(all_ys) - min(all_ys),
    max(all_zs) - min(all_zs)
]
max_range = max(ranges)
mean_xs = sum(all_xs) / len(all_xs)
mean_ys = sum(all_ys) / len(all_ys)
mean_zs = sum(all_zs) / len(all_zs)

# Set the range and tick labels for each axis
ax.set_xticks([mean_xs - max_range/2, mean_xs, mean_xs + max_range/2])
ax.set_xlim(mean_xs - max_range/2, mean_xs + max_range/2)
ax.set_xticklabels(['{:.1f} m'.format(mean_xs - max_range/2), '{:.1f} m'.format(mean_xs), '{:.1f} m'.format(mean_xs + max_range/2)])
ax.set_xlabel('X position (m)')

ax.set_yticks([mean_ys - max_range/2, mean_ys, mean_ys + max_range/2])
ax.set_ylim(mean_ys - max_range/2, mean_ys + max_range/2)
ax.set_yticklabels(['{:.1f} m'.format(mean_ys - max_range/2), '{:.1f} m'.format(mean_ys), '{:.1f} m'.format(mean_ys + max_range/2)])
ax.set_ylabel('Y position (m)')

ax.set_zticks([mean_zs - max_range/2, mean_zs, mean_zs + max_range/2])
ax.set_zlim(mean_zs - max_range/2, mean_zs + max_range/2)
ax.set_zticklabels(['{:.1f} m'.format(mean_zs - max_range/2), '{:.1f} m'.format(mean_zs), '{:.1f} m'.format(mean_zs + max_range/2)])
ax.set_zlabel('Z position (m)')

# Set the plot title and legend
if connect_points:
    plt.title("Ground Truth Path Trajectory")
else:
    plt.title("Trajectory")
plt.legend()

# Show the plot
plt.show()