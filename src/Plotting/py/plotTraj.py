import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# List of up to 4 file paths to text files
# file_paths = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]
file_paths = ["GTcamTrajectory.txt", "camTrajectory.txt"]

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

    # Plot the x, y, z values as a scatter plot
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, label=file_path)

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
ax.set_xlim(mean_xs - max_range/2, mean_xs + max_range/2)
ax.set_xticklabels(['x={:.1f}'.format(mean_xs - max_range/2), 'x={:.1f}'.format(mean_xs), 'x={:.1f}'.format(mean_xs + max_range/2)])
ax.set_ylim(mean_ys - max_range/2, mean_ys + max_range/2)
ax.set_yticklabels(['y={:.1f}'.format(mean_ys - max_range/2), 'y={:.1f}'.format(mean_ys), 'y={:.1f}'.format(mean_ys + max_range/2)])
ax.set_zlim(mean_zs - max_range/2, mean_zs + max_range/2)
ax.set_zticklabels(['z={:.1f}'.format(mean_zs - max_range/2), 'z={:.1f}'.format(mean_zs), 'z={:.1f}'.format(mean_zs + max_range/2)])

# Set the plot title and legend
plt.title("3D Plot of X,Y,Z Values from Text Files")
plt.legend()

# Show the plot
plt.show()