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


# class that has both the path and the array of the tif file
class tif_array:
    def __init__(self, path,imarray):
        self.path = path
        self.imarray = imarray

class tree_pose:
    def __init__(self, x, y, z):
      self.x = x
      self.y = y
      self.z = z

# Get Path of tif file
def get_path(path):
  splitted_path = path.split('/')
  return Path(__file__).parents[1] / splitted_path[0] / splitted_path[1] / splitted_path[2]

# Get the tif file as a numpy array
def get_heightmap_array(path_to_tif):
  return np.array(Image.open(path_to_tif))    

def tif_info(path):
  # Get Image and Convert it to np.array
  im = Image.open(path)
  imarray = np.array(im)
  print('shape of tif array ' +str(imarray.shape))
  print('/n' + str(imarray))
  return imarray

def create_xml(tree_model):
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

    {tree_model}
  </world>
</sdf>
    '''
    return xml_string

def create_models_string(tree_name, previous_string, dif,pose):
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
      return tree_models + previous_string

def write_to_file(string,file):
    file_path = str(Path(__file__).parents[0] / file)
    f = open(file_path,"w")
    f.write(string)
    print("Written xml String to file : " + file_path )
    f.close()

def create_dae_name_array(tree_number):
      tree_array = np.array(["Tree_" + str(tree_number) + "_Trunk"])
      return np.append(tree_array,["Tree_" + str(tree_number) + "_Leaf_" + str(i) for i in range(1,4)])

def create_models_xml(start_x,start_y,finish_x,finish_y,heightmap):
  tree_models =''''''
  counter = 0
  for x in range(start_x,finish_x,2):
        for y in range(start_y,finish_y,2):
              z = heightmap[int(np.round((x+40)*129/80,0))][int(np.round((y+40)*129/80,0))] - 0.2
              pose = tree_pose(x,y,z)
              tree_models = create_models_string(create_dae_name_array(random.randint(1,4)), tree_models,counter,pose)
              counter += 1
  return tree_models


def main():
      
  path = get_path('worlds/tif_files/cropped_gdal.tif')
  imarray = get_heightmap_array(path)
  tif_obj = tif_array(path,imarray)
  # Edit array (minimum of array should be 0.0) and round the elements
  tif_obj.imarray = np.around(tif_obj.imarray - np.min(tif_obj.imarray),decimals=2)
  # Max Height should be 10m so we Normalize the array is the maximum value being 10
  tif_obj.imarray = 10 * tif_obj.imarray / np.max(tif_obj.imarray)
  # imarray needs to be rotated for the right values according to the heightmap
  tree_models = create_models_xml(start_x=-41,start_y=-41,finish_x=31,finish_y=31,heightmap=np.rot90(tif_obj.imarray,3))
  xml_string = create_xml(tree_models)
  write_to_file(xml_string,"trial.world")



if __name__ == '__main__':
      main()
    

