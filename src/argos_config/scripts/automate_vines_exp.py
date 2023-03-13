#!/usr/bin/python3
# This script is used to add vineyards to a Gazebo World
# It get the slope information from the heightmap (tif file)
# Changes an xml file that describes the Gazebo World

from json.tool import main
import string
from typing_extensions import Self
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
import random

def create_xml():
  xml_string = f'''<?xml version="1.0" ?>
  <sdf version="1.7">
  <world name="default">
  <physics type="ode">
      <ode>
        <solver>
          <type>world</type>
        </solver>
        <constraints>
          <contact_max_correcting_vel>0.1</contact_max_correcting_vel>
          <contact_surface_layer>0.0001</contact_surface_layer>
        </constraints>
      </ode>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>
    <gravity>0.0 0.0 -9.81</gravity>
  <!-- A global light source -->
    <light name='sun2' type='point'>
      <cast_shadows>1</cast_shadows>
      <pose frame=''>-0.5 2 5 0 -0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>100000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>0 0 1</direction>
    </light>
    <gui fullscreen='0'>
      <camera name='user_camera'>
        <pose>-0.059361 2.171367 15.466796 -0.000711 1.569796 1.571162</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>

  '''
  return xml_string

def create_model_dae(obj_name, obj_mat,counter, pose, rot, scale, shadows):
  obj = f'''
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>30 30</size>
        </plane>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>30 30</size>
        </plane>
      </geometry>
      <material>
      <script>
        <uri>model://../../mini_project/vines/materials/scripts/</uri>
        <uri>model://../../mini_project/vines/materials/textures/</uri>
        <name>Plane</name>
      </script>
      </material>
    </visual>
  </link>
</model>
  
  '''
  return obj

def create_apriltag_model(obj_mat,counter, pose, rot):
  obj = f'''
  <model name='Apriltag36_11_00000_{counter}'>
    <link name='main'>
    <pose frame=''>{pose.x} {pose.y} {pose.z} {rot.x} {rot.y} 0</pose>
      <visual name='main_Visual'>
        <geometry>
          <box>
          <size>0.6 0.6 0.0001</size>
          </box>
        </geometry>
           <material>
          <script>
            <uri>model://../../mini_project/vines/materials/scripts/</uri>
            <uri>model://../../mini_project/vines/materials/textures/</uri>
            <name>Obj/{obj_mat}</name>
          </script>
        </material>
      </visual>
     </link>
    <static>1</static>
  </model>
  
  '''
  return obj

def create_models_string(tree_name, dif,pose,model):
  if model < 4:
    tree_models = f'''
    <model name='tree_{dif}'>
    <static>true</static>
    <link name='link'>
      <visual name='trunk'>
        <geometry>
          <mesh>
            <uri>model://../../mini_project/vines/meshes/{tree_name[0]}.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <uri>model://../../mini_project/vines/materials/scripts/</uri>
            <uri>model://../../mini_project/vines/materials/textures/</uri>
            <name>Vine/Bark</name>
          </script>
        </material>
      </visual>
      <visual name='leafa'>
        <geometry>
          <mesh>
            <uri>model://../../mini_project/vines/meshes/{tree_name[1]}.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <uri>model://../../mini_project/vines/materials/scripts/</uri>
            <uri>model://../../mini_project/vines/materials/textures/</uri>
            <name>Vine/Leafa</name>
          </script>
        </material>
      </visual>
      <visual name='leafb'>
        <geometry>
          <mesh>
            <uri>model://../../mini_project/vines/meshes/{tree_name[2]}.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <uri>model://../../mini_project/vines/materials/scripts/</uri>
            <uri>model://../../mini_project/vines/materials/textures/</uri>
            <name>Vine/Leafb</name>
          </script>
        </material>
      </visual>
      <visual name='leafc'>
        <geometry>
          <mesh>
            <uri>model://../../mini_project/vines/meshes/{tree_name[3]}.dae</uri>
          </mesh>
        </geometry>
        <material>
          <script>
            <uri>model://../../mini_project/vines/materials/scripts/</uri>
            <uri>model://../../mini_project/vines/materials/textures/</uri>
            <name>Vine/Leafc</name>
          </script>
        </material>
      </visual>
      <self_collide>0</self_collide>
      <enable_wind>0</enable_wind>
      <kinematic>0</kinematic>
    </link>
    <pose>{pose.x} {pose.y} {pose.z} 0 0 0</pose>
  </model>
    
    '''
  elif model >= 4:
      tree_models = f'''
      
      '''
  return tree_models

class tree_pose:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

class xml_file_class:
  def __init__(self, path):
    self.path = path

  # initializing the file by overwriting 
  def initialize_file(self):
    f = open(self.path,"w")
    f.write(create_xml())
    print("Initialized file  : " + str(self.path) )
    f.close()

  # adding to the file without overwriting
  def write_to_file(self,string):
    f = open(self.path,"a")
    f.write(string)
    f.close()

  def print_path(self):
    return str(self.path)

# Get Path of tif file
def get_path(path):
  splitted_path = path.split('/')
  return Path(__file__).parents[1] / splitted_path[0] / splitted_path[1] / splitted_path[2]

# Get the tif file as a numpy array
def get_heightmap_array(path_to_tif):
  return np.array(Image.open(path_to_tif))   
  
# create an array of strings that contain name of the dae models according to the random number generated
def create_dae_name_array(tree_number):
  tree_array = np.array(["Tree_" + str(tree_number) + "_Trunk"])
  return np.append(tree_array,["Tree_" + str(tree_number) + "_Leaf_" + str(i) for i in range(1,4)])

# The main function that with each loop adds a new vine
def create_models_xml(start_x,start_y,finish_x,finish_y,counter,xml):
  model_number = 0
  for x in range(start_x,finish_x,2):
    for y in range(start_y,finish_y,1):
      if x==1 and y == 1:
          continue
      # Check out of bounds
      pose = tree_pose(x,y,-0.3)
      # If more models are added to take one randomly
      # model_number = random.randint(0,4)
      if model_number < 4:
        tree_model = create_models_string(create_dae_name_array(1),counter,pose,model_number)
      elif model_number >= 4:
        tree_model = create_models_string(create_dae_name_array(1),counter,pose,model_number)
      xml.write_to_file(tree_model)
      counter += 1

  return counter

def main():
      
  world_file = str(Path(__file__).parents[1] / "worlds" / "vineyard" / "vineyard_2.world")
  # Initialize the world file to contain the heightmap 
  xml_file = xml_file_class(world_file)
  xml_file.initialize_file()

  # Create the array that contains the heightmap values (so that the vines have correct height)
  path = get_path('worlds/tif_files/cropped_gdal.tif')
  tifarray = get_heightmap_array(path)
  # Edit array (minimum of array should be 0.0) and round the elements
  tifarray = np.around(tifarray - np.min(tifarray),decimals=2)
  # Max Height should be 10m so we set the maximum value of the array as 10
  tifarray = 10 * tifarray / np.max(tifarray)


  # tifarray needs to be rotated for to get the correct values according to the heightmap
  counter = 0
  counter = create_models_xml(start_x=-5,start_y=-3,finish_x=7,finish_y=10,counter = counter,xml=xml_file)
  obj_model = create_model_dae(obj_name="Floor", obj_mat="Floor",counter = counter, pose = tree_pose(-4,-8,0.03), rot = 0,scale = 10, shadows=1)
  xml_file.write_to_file(obj_model)
  counter += 1
  obj_model = create_apriltag_model(obj_mat="Apriltag36_11_00000",counter = counter, pose = tree_pose(1,1,0.4), rot = tree_pose(0,1.57079632679,0))
  xml_file.write_to_file(obj_model)
  counter += 1
  
  end_string = '''
    </world>
  </sdf>
  '''
  print("Written End String to file : " + xml_file.print_path())
  xml_file.write_to_file(end_string) 
  print("World File Ready : " + world_file)



if __name__ == '__main__':
  main()
    

