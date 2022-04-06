# This script is used to add vineyards to a Gazebo World
# It get the slope information from the heightmap (tif file)
# Changes an xml file that describes the Gazebo World

from json.tool import main
import string
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path


# Get Path of tif file
def get_path():
    cwd = Path.cwd()
    return cwd.parent.absolute() / 'worlds' / 'tif_files' / 'cropped_gdal.tif'

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

  </world>
</sdf>
    '''
    return xml_string

def modify_file(string):

    f = open("trial.world","w")
    f.write(string)
    f.close()

def main():
    path = get_path()
    print(path)
    tif_array = tif_info(path)
    xml_string = create_xml()
    modify_file(xml_string)




if __name__ == '__main__':
    main()
    

