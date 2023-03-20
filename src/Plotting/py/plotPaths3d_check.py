import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read in the data from the first text file
data1 = []
with open("ground_truth_traj.txt", "r") as f:
    for line in f:
        data1.append([float(x) for x in line.strip().split()])

# Read in the data from the second text file
data2 = []
with open("dual_cam_traj_exp_07.txt", "r") as f:
    for line in f:
        data2.append([float(x) for x in line.strip().split()])

# Extract the columns of interest from the first file
x1 = [row[3] for row in data1]
y1 = [row[7] for row in data1]
z1 = [row[11] for row in data1]

# Extract the columns of interest from the second file
x2 = [row[3] for row in data2]
y2 = [row[7] for row in data2]
z2 = [row[11] for row in data2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, label='Data 1', s=1)
ax.scatter(x2, y2, z2, label='Data 2', s=1)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

# Add a legend
ax.legend()

# Show the plot
plt.show()