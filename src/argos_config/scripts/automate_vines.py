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


# class that has both the path and the array of the tif file
class tif_array:
    def __init__(self, path,imarray):
        self.path = path
        self.imarray = imarray

# Get Path of tif file
def get_path(path):
  splitted_path = path.split('/')
  return Path(__file__).parents[1] / splitted_path[0] / splitted_path[1] / splitted_path[2]

# Get the tif file as a numpy array
def get_heightmap_array(path_to_tif):
  return np.array(Image.open(path_to_tif))    

class xml_file:
  path = Path(__file__).parents[1] / 'worlds' / 'tif_files' / 'cropped_gdal.tif'




def tif_info(path):
  # Get Image and Convert it to np.array
  im = Image.open(path)
  imarray = np.array(im)
  print('shape of tif array ' +str(imarray.shape))
  print('/n' + str(imarray))
  return imarray

def create_xml():
    xml_string = '''<sdf version="1.7">
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
              <pos>0 0 -320</pos>
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
              <pos>0 0 -320</pos>
            </heightmap>
          </geometry>
        </visual>

      </link>
    </model>

    <model name='texture_1'>
      <static>1</static>
      <link name='link'>
        <visual name='visual'>
          <geometry>
            <mesh>
              <uri>model://../../mini_project/meshes/vines.stl</uri>
              <scale>0.03 0.03 0.03</scale>
            </mesh>
          </geometry>
        </visual>
        <collision name='right-angle'>
          <geometry>
            <mesh>
              <uri>model://../../mini_project/meshes/vines.stl</uri>
              <scale>0.03 0.03 0.03</scale>
            </mesh>
          </geometry>
          <max_contacts>10</max_contacts>
          <surface>
            <contact>
              <ode/>
            </contact>
            <bounce/>
            <friction>
              <torsional>
                <ode/>
              </torsional>
              <ode/>
            </friction>
          </surface>
        </collision>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
      <pose>0 2 1 0 -3.13</pose>
    </model>

  </world>
</sdf>
    '''
    return xml_string

def write_to_file(string,file):
    file_path = str(Path(__file__).parents[0] / file)
    f = open(file_path,"w")
    f.write(string)
    print("Written xml String to file : " + file_path )
    f.close()

def main():
      
  path = get_path('worlds/tif_files/cropped_gdal.tif')
  imarray = get_heightmap_array(path)
  tif_obj = tif_array(path,imarray)
  # Edit array (minimum of array should be 0.0) and round the elements
  tif_obj.imarray = np.around(tif_obj.imarray - np.min(tif_obj.imarray),decimals=2)
  # Max Height should be 10m so we Normalize the array is the maximum value being 10
  tif_obj.imarray = 10 * tif_obj.imarray / np.max(tif_obj.imarray)
  xml_string = create_xml()
  write_to_file(xml_string,"trial.world")




if __name__ == '__main__':
      main()
    

