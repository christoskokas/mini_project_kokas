import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import string

def getData(fileName,sep) :
    absolute_path = os.path.dirname(__file__)
    relative_path = "/../"
    full_path = os.path.join(absolute_path, relative_path)
    path = Path(absolute_path)

    path2 = path.parent.absolute()
    datapath = str(path2) + "/data/" + fileName

    with open(datapath) as f:
        # w, h = [float(x) for x in next(f).split()]
        array = [[float(x) for x in line.split(sep)] for line in f]    
    return np.array(array)

def getKitti(fileName, size) :
    absolute_path = os.path.dirname(__file__)
    relative_path = "/../"
    full_path = os.path.join(absolute_path, relative_path)
    path = Path(absolute_path)

    path2 = path.parent.absolute()
    datapath = str(path2) + "/data/" + fileName
    arrayX = np.zeros(shape = (size,1))
    arrayY = np.zeros(shape = (size,1))
    arrayZ = np.zeros(shape = (size,1))
    arrayR = np.zeros(shape = (size,3,3))
    with open(datapath) as f:
        # [[float(x) for x in line.split()] for line in f]
        count = 0
        for line in f:
            i = 1
            for x in line.split() :
                if (i == 4) :
                    arrayX[count,0] = x
                elif (i == 8) :
                    arrayZ[count,0] = x
                elif (i == 12):
                    arrayY[count,0] = x
                else :
                    arrayR[count,0,0] = x
                i += 1
            count += 1
    return arrayX,arrayY,arrayZ,arrayR

x,y,z,r = getKitti("00.txt", 4541)
xp,yp,zp,rp = getKitti("zedPoses.txt", 4541)

ex = x - xp
ey = y - yp
ez = z - zp


# print(x)
# print(y)
# print(z)
fig = plt.figure(1)
ax1 = fig.add_subplot(projection='3d')

ax1.plot(x, y, z)
ax1.plot(xp, yp, zp, 'r')

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.set_xlim(-600,600)
ax1.set_ylim(-600,600)
ax1.set_zlim(-600,600)


fig2 = plt.figure(2)
axer1 = fig2.add_subplot(3,1,1)
axer1.plot(ex)
axer1.set_xlabel('frames')
axer1.set_ylabel('Error X in meters')

axer1 = fig2.add_subplot(3,1,2)
axer1.plot(ey)
axer1.set_xlabel('frames')
axer1.set_ylabel('Error Y in meters')

axer1 = fig2.add_subplot(3,1,3)
axer1.plot(ez)
axer1.set_xlabel('frames')
axer1.set_ylabel('Error Z in meters')

plt.show()
# array = getData("data.txt",',')
# print(array)

# array2 = getData("data2.txt",' ')
# print(array2)

# array3 = getData("data3.txt",' ')
# print(array3)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(array, array2, array3)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

# fig, ax = plt.subplots()

# fruits = ['apple', 'blueberry', 'cherry', 'orange']
# counts = [40, 100, 30, 55]
# bar_labels = ['red', 'blue', '_red', 'orange']
# bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

# ax.set_ylabel('fruit supply')
# ax.set_title('Fruit supply by kind and color')
# ax.legend(title='Fruit color')

# plt.show()