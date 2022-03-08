import numpy as np

import pyvista as pv
from pyvista import examples
import open3d as o3d
import pathlib
import os

# Define some helpers - ignore these and use your own data!
def generate_points(subset=0.02):
    """A helper to make a 3D NumPy array of points (n_points by 3)"""
    dataset = examples.download_lidar()
    ids = np.random.randint(low=0, high= dataset.n_points-1,
                            size=(10, 10, 10))
    return dataset.points[ids]

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

points = ids = np.random.randint(low=0, high= 100,
                            size=(100, 9))
# Print first 5 rows to prove its a numpy array (n_points by 3)
# Columns are (X Y Z)
print(type(points))

#point_cloud = pv.PolyData(points)
#print(type(point_cloud))
#print(np.allclose(points, point_cloud.points))
#point_cloud.plot(eye_dome_lighting=True)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(points[:,3:6]/255)
pcd.normals = o3d.utility.Vector3dVector(points[:,6:9])
#o3d.visualization.draw_geometries([pcd])

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
dec_mesh = mesh.simplify_quadric_decimation(100000)

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

#Get Absolute Path
path=pathlib.Path(__file__).parent.resolve()

#my_lods = lod_mesh_export(dec_mesh, [100000,50000,10000,1000,100], ".stl", str(path)+"/trial_point")


#o3d.visualization.draw_geometries([my_lods[1000]])


poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)







o3d.io.write_triangle_mesh(str(path)+"/bpa_mesh.stl", dec_mesh)
o3d.io.write_triangle_mesh(str(path)+"/p_mesh_c.stl", p_mesh_crop)


#my_lods2 = lod_mesh_export(p_mesh_crop, [8000,800,300], ".stl", str(path)+"/trial_point_poisson")

#o3d.visualization.draw_geometries([my_lods2[8000]])
