from stl import mesh
import math
import numpy
import random
import pathlib
import os

def append_to_file(path):
    if (os.path.exists(str(path)+"/seeds.txt")):
        with open(str(path)+"/seeds.txt", "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")
            # Append text at the end of file
            file_object.write("The selected seed is : " + str(seed))
            file_object.flush()
            file_object.close()
            print("Seed written to file seeds.txt")
    else:
        print("No File seeds.txt in path")  



    
#Get Absolute Path
path=pathlib.Path(__file__).parent.resolve()

seed = random.randint(0,100000)
print("The selected seed is : ", seed)


append_to_file(path)

data = numpy.zeros(2, dtype=mesh.Mesh.dtype)

# Top of the cube 
# x,y,z lines
data['vectors'][0] = numpy.array([[0, 2, 1],
                                  [1, 0, 3],
                                  [0, 0, 1]])
data['vectors'][1] = numpy.array([[0, 2, 1],
                                  [1, 0, 3],
                                  [0, 3, 4]])
# Optionally render the rotated cube faces

your_mesh = mesh.Mesh(data, remove_empty_areas=False)

from matplotlib import pyplot
from mpl_toolkits import mplot3d

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

your_mesh.save('path/trial.stl')
# Show the plot to the screen
pyplot.show()