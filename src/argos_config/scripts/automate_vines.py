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
  <!-- A global light source -->
  <include>
    <uri>model://sun</uri>
  </include>

  <model name="heightmap">
    <static>true</static>
    <link name="heightmap_argos">
      <collision name="collision">
        <geometry>
          <heightmap>
            <uri>model://tif_files/cropped_gdal.tif</uri>
            <size>80 80 10</size>
            <pos>0 0 -318.28</pos>
          </heightmap>
        </geometry>
      </collision>

      <visual name="visual_abcedf">
        <geometry>
          <heightmap>
            <texture>
              <diffuse>file://media/materials/textures/dirt_diffusespecular.png</diffuse>
              <normal>file://media/materials/textures/flat_normal.png</normal>
              <size>1</size>
            </texture>
            <texture>
              <diffuse>file://media/materials/textures/grass_diffusespecular.png</diffuse>
              <normal>file://media/materials/textures/flat_normal.png</normal>
              <size>1</size>
            </texture>
            <texture>
              <diffuse>file://media/materials/textures/fungus_diffusespecular.png</diffuse>
              <normal>file://media/materials/textures/flat_normal.png</normal>
              <size>1</size>
            </texture>
            <blend>
              <min_height>2</min_height>
              <fade_dist>5</fade_dist>
            </blend>
            <blend>
              <min_height>4</min_height>
              <fade_dist>5</fade_dist>
            </blend>
            <uri>model://tif_files/cropped_gdal.tif</uri>
            <size>80 80 10</size>
            <pos>0 0 -318.28</pos>
          </heightmap>
        </geometry>
      </visual>

    </link>
  </model>

  '''
  return xml_string

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
      <collision name='tree_col'>
        <pose>0 0 0.5 0 0 0</pose>
        <geometry>
          <box>
            <size>0.3 0.3 1 </size>
          </box>
        </geometry>
      </collision>
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
def create_models_xml(start_x,start_y,finish_x,finish_y,heightmap,xml):
  counter = 0
  model_number = 0
  for x in range(start_x,finish_x,2):
    for y in range(start_y,finish_y,2):
      # Check out of bounds
      if (int(np.round((x+40)*129/80,0)) < 0 or int(np.round((x+40)*129/80,0)) > len(heightmap) or int(np.round((y+40)*129/80,0)) < 0 or int(np.round((y+40)*129/80,0)) > len(heightmap) ):
        continue

      
      z = heightmap[int(np.round((x+40)*129/80,0))][int(np.round((y+40)*129/80,0))] - 0.2
      pose = tree_pose(x,y,z)
      # If more models are added to take one randomly
      # model_number = random.randint(0,4)
      if model_number < 4:
        tree_model = create_models_string(create_dae_name_array(random.randint(1,4)),counter,pose,model_number)
      elif model_number >= 4:
        tree_model = create_models_string(create_dae_name_array(random.randint(1,4)),counter,pose,model_number)
      xml.write_to_file(tree_model)
      counter += 1
  end_string = '''
    </world>
  </sdf>
  '''
  print("Written End String to file : " + xml.print_path())
  xml.write_to_file(end_string) 

def main():
      
  world_file = str(Path(__file__).parents[1] / "worlds" / "vineyard" / "vineyard.world")
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
  create_models_xml(start_x=-5,start_y=-11,finish_x=15,finish_y=11,heightmap=np.rot90(tifarray,3),xml=xml_file)
  print("World File Ready : " + world_file)



if __name__ == '__main__':
  main()
    

