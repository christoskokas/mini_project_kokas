import numpy as np

# Specify the file paths
file1_path = 'ground_truth_traj_real_for_change.txt'
file2_path = 'dual_cam_real_for_change.txt'

# Load the data from the two files
file1_data = np.loadtxt(file1_path)
file2_data = np.loadtxt(file2_path)

# Extract the relevant columns from the two files
file1_xyz = file1_data[:, [7, 11, 3]]
file2_xyz = file2_data[:, [3, 11, 7]]
file1_xyz[:, 0] = -file1_xyz[:, 0]

# Add the columns to get the desired output
result_xyz = file1_xyz
# print(file1_xyz)
# print(file2_xyz)
# Calculate the average difference between file1 and file2 for every 10 rows
diffs = []
for i in range(0, len(file1_data), 10):
    avg_diff = np.mean(file2_data[i:i+10, 11] - file1_data[i:i+10, 11])
    diffs.extend([-0.05]*10)
diffs = diffs[:-1]
# Add the differences to the z column of the first file
result_xyz[:, 1] += np.array(diffs)
print(diffs[1])

# Combine the original columns with the new results and save to a new file
result_data = file1_data
result_data[:,11] += np.array(diffs)
np.savetxt('result_gt.txt', result_data)