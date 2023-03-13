import matplotlib.pyplot as plt

# Define the coordinates of the root of the tree
x0, y0 = 0, 0

# Define the size of the tree
size = 10

# Define the coordinates of the branches
x1, y1 = x0 - size/2, y0 + size/2
x2, y2 = x0 + size/2, y0 + size/2
x3, y3 = x0 - size/4, y0 + size

# Draw the circles and lines to form the tree
plt.scatter(x0, y0, s=500, facecolors='none', edgecolors='brown', linewidth=3)
plt.scatter(x1, y1, s=250, facecolors='green', edgecolors='none')
plt.scatter(x2, y2, s=250, facecolors='green', edgecolors='none')
plt.scatter(x3, y3, s=250, facecolors='green', edgecolors='none')
plt.plot([x0, x1], [y0, y1], color='brown', linewidth=5)
plt.plot([x0, x2], [y0, y2], color='brown', linewidth=5)
plt.plot([x0, x3], [y0, y3], color='brown', linewidth=5)

# Set the limits of the plot
plt.xlim(-size, size)
plt.ylim(0, size*2)

# Hide the axes
plt.axis('off')

# Show the plot
plt.show()