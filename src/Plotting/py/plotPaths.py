import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as patches

# List of up to 4 file paths to text files
# file_paths = ["ground_truth_obj.txt","dual_cam_obj.txt", "orbslam3_obj.txt"]
# file_paths = ["ground_truth_big.txt","dual_cam_big_LC.txt"]
# file_paths = ["ground_truth_vines.txt"]
# file_paths = ["ground_truth_traj.txt", "dual_cam_traj_exp_07.txt", "orbslam3_exp_07.txt"]
# file_paths = ["ground_truth_traj.txt", "dual_cam_traj_exp_07.txt"]

# file_paths = ["empty.txt", "dual_cam_traj_exp_07.txt", "orbslam3_exp_07.txt"]
# file_paths = ["ground_truth_traj_exp_07.txt"]
file_paths = ["ground_truth_traj_exp_07.txt", "dual_cam_traj_exp_07.txt", "orbslam3_exp_07.txt"]
# file_paths = ["simu_ground_truth_big.txt","simu_dual_cam_big.txt"]
# file_paths = ["simu_ground_truth_small.txt","simu_dual_cam_LC_small.txt","simu_dual_cam_NLC_small.txt","orbslam3_LC.txt"]
# file_paths = ["ground_truth_vines.txt","simu_dual_cam_vines.txt","simu_orbslam3_vines.txt"]

# file_paths = ["result_gt.txt", "dual_cam_real_for_change.txt","orbslam3_cls.txt"]
# file_paths = ["ground_truth_traj.txt", "dual_cam_exp_6.txt"]
# file_paths = ["ground_truth_vines.txt","dual_cam_vines.txt", "orbslam3_vines.txt"]
# file_paths = ["ground_truth_LC.txt","dual_cam_LC.txt","dual_cam_NLC.txt","simu_orbslam3_small.txt"]
# file_paths = ["ground_truth_LC.txt","dual_cam_LC.txt","dual_cam_NLC.txt","orbslam3_LC.txt"]
# file_paths = ["ground_truth_L.txt","dual_cam_L.txt", "orbslam3.txt"]
home_path = "/home/christos/catkin_ws/src/mini_project_kokas/src/vio_slam/build/devel/lib/vio_slam"
# file_paths = [home_path + "/ground_truth_traj.txt"]
# file_paths = [home_path + "/ground_truth_traj.txt", home_path + "/single_cam_traj.txt", home_path + "/ORB-SLAM3_traj.txt"]
# file_paths = [home_path + "/ground_truth_traj_exp2.txt", home_path + "/single_cam_traj_exp2.txt", home_path + "/ORB-SLAM3_traj_exp2.txt"]


colors = ['b', 'r', 'g','y']

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
        if i==0:
            xs.append(-float(values[7]))
            ys.append(float(values[11]))
            zs.append(float(values[11]))
        else:
            xs.append(float(values[3]))
            ys.append(float(values[11]))
            zs.append(float(values[7]))

        # if i==0:
        #     xs.append(float(values[11]))
        #     ys.append(-float(values[3]))
        #     zs.append(float(values[11]))
        # else:
        #     xs.append(float(values[11]))
        #     ys.append(-float(values[3]))
        #     zs.append(float(values[7]))
            
            
        # if i == 0 :
        # else : 
        #     zs.append(-float(values[7]))

    # Store the x, y, z values in the overall lists
    all_xs.extend(xs)
    all_ys.extend(ys)
    all_zs.extend(zs)

    # Get the file name without the .txt extension
    file_name = file_path.split(".")[0]
    filename = os.path.basename(file_path)
    base_name, extension = os.path.splitext(filename)

    # if i == 0:
    #     plt.plot(xs[0], ys[0], marker="o", markersize = 10, markerfacecolor="green",markeredgecolor='none', label='start', linestyle='None')
    #     plt.plot(xs[-1], ys[-1], marker="o", markersize = 10, markerfacecolor="black",markeredgecolor='none', label='end', linestyle='None')

    # Plot the x, y, z values as a scatter plot with optional lines connecting the points
    if connect_points:
        if i == 0 :
            plt.plot(xs, ys, colors[i], label="Ground Truth", linewidth = 1)
        elif i == 1 :
            plt.plot(xs, ys, colors[i], label="Dual Cam LC", linewidth = 1)
        elif i == 2 :
            plt.plot(xs, ys, colors[i+1], label="ORB-SLAM3", linewidth = 1)
        else : 
            plt.plot(xs, ys, colors[i], label="ORB-SLAM3", linewidth = 1)
            
    else:
        plt.scatter(xs, ys, s=1, label=base_name, color = colors[i])

    # if i == 2:
    #     plt.annotate('ORB-SLAM3 resets here', xy=(xs[-1],ys[-1]), xytext=(-0.2, ys[-1]), arrowprops=dict(facecolor='red', shrink=0.03))

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    if i ==len(file_paths) - 1 :
        plt.plot([], [], ' ', label="Featureless\n Object", marker='s', markersize=10, markerfacecolor='none', markeredgecolor='b', mew=2)
        # plt.plot([], [], ' ', label="Grapevines", marker='s', markersize=10, markerfacecolor='g', markeredgecolor='tab:brown', mew=2)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.locator_params(axis='y',nbins=5)
    # plt.locator_params(axis='x',nbins=5)
    plt.title("Experiment Trajectories")
    # plt.ylim(-3, 5.5)
    # plt.xlim(-7, 10)
    # plt.xlim(-3, 5.5)
    # plt.xlim(-2, 2)
    plt.xlim(-0.3, 0.65)
    plt.tight_layout()
    legend = plt.legend(loc='upper right')
    # legend.get_frame().set_alpha(0.5)
    plt.grid(True)
    
    # ax = plt.axes()
    # ax.set_facecolor("grey")
    # Add a sphere symbol of green color at the start of the trajectory and a sphere symbol of red color at the end

    if i == 0:
        prevx = -5
        prevy = -5
        for j in range(50, len(xs)-5, 1):
            # if j+40 < len(xs):
            dx = xs[j+1] - xs[j]
            dy = ys[j+1] - ys[j]
            if abs(prevx - xs[j]) > 0.1 or abs(prevy - ys[j]) > 0.1 :
                # plt.arrow(xs[j],ys[j],dx,dy, width = 0.03, color = colors[i])
                # plt.arrow(xs[j],ys[j],dx,dy, width = 0.005, color = colors[i])
                prevx = xs[j]
                prevy = ys[j]
                
plt.show()
# plt.savefig('vine_trans_2.png', transparent = True)
# plt.savefig('vine_t.png', format='eps')
                # ax.quiver(xs[j], ys[j], dx, dy, length=0.01, arrow_length_ratio=10, normalize=True, color=colors[i])

    # Add arrows on each trajectory that show the heading of the trajectory
    # if i == 0:
    #     xdir = np.diff(xs)
    #     ydir = np.diff(ys)
    #     zdir = np.diff(zs)
    #     ax.quiver(xs[:-1], ys[:-1], zs[:-1], xdir, ydir, zdir, color = colors[i], arrow_length_ratio=2)

# Calculate the range and mean of all x, y, z values
# ranges = [
#     max(all_xs) - min(all_xs),
#     max(all_ys) - min(all_ys),
#     max(all_zs) - min(all_zs)
# ]
# max_range = max(ranges)
# mean_xs = sum(all_xs) / len(all_xs)
# mean_ys = sum(all_ys) / len(all_ys)
# mean_zs = sum(all_zs) / len(all_zs)

# # Set the range and tick labels for each axis
# ax.set_xticks([mean_xs - max_range/2, mean_xs, mean_xs + max_range/2])
# ax.set_xlim(mean_xs - max_range/2, mean_xs + max_range/2)
# ax.set_xticklabels(['{:.1f} m'.format(mean_xs - max_range/2), '{:.1f} m'.format(mean_xs), '{:.1f} m'.format(mean_xs + max_range/2)])
# ax.set_xlabel('X (m)')

# ax.set_yticks([mean_ys - max_range/2, mean_ys, mean_ys + max_range/2])
# ax.set_ylim(mean_ys - max_range/2, mean_ys + max_range/2)
# ax.set_yticklabels(['{:.1f} m'.format(mean_ys - max_range/2), '{:.1f} m'.format(mean_ys), '{:.1f} m'.format(mean_ys + max_range/2)])
# ax.set_ylabel('Y (m)')

# ax.set_zticks([mean_zs - max_range/2, mean_zs, mean_zs + max_range/2])
# ax.set_zlim(-1,1)
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")
# # ax.set_zticklabels(['{:.1f} m'.format(mean_zs - max_range/2), '{:.1f} m'.format(mean_zs), '{:.1f} m'.format(mean_zs + max_range/2)])
# ax.set_zlabel('Z (m)')

# ax.grid(b=True, which='major', axis='both', linestyle='-')

# Set the plot title and legend
# if connect_points:
#     plt.title("Simulation Trajectories")
# else:
#     plt.title("Simulation Trajectories")
# plt.legend()

# Show the plot
# plt.show()