import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_data(files):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, file in enumerate(files):
        with open(file, 'r') as f:
            data = f.read().splitlines()

        x = []
        y = []
        z = []

        for line in data:
            values = line.split()
            x.append(float(values[0]))
            y.append(float(values[1]))
            z.append(float(values[2]))

        ax.scatter(x, y, z, label=f'File {i+1}')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    files = ['camPosition.txt', 'GTcamPosition.txt']
    # files = ['camPosition.txt', 'GTcamPosition.txt', 'data3.txt', 'data4.txt']
    plot_3d_data(files)