# mini_project_kokas

Goal of this project is the modelling of the argos quadruped robot using champ setup assistant. Equipped with a stereo camera and by using gmapping ROS packages, argos can move and avoid obstacles. To avoid obstacles, packages that convert the images from the stereo camera to laserscan are used.

## Ιnstallation

Clone the packages needed on catkin workspace in the src folder or on a git folder and make a symlink to the catkin workspace. 

```console
git clone --recursive https://github.com/christoskokas/mini_project_kokas.git
```

From the catkin workspace install all the needed dependencies for the packages.

```console
rosdep install --from-paths src --ignore-src -r -y
```

Build the workspace.

```console
catkin build argos_config
catkin build pointcloud_to_laserscan -DCMAKE_BUILD_TYPE=Release 
source devel/setup.bash
```

## Quick Start

### Disparity to Laserscan 

Run the Gazebo environment:

```console
roslaunch argos_config argos_gazebo.launch
```

On another terminal run:

```console
roslaunch argos_config slam.launch
```

### PointCloud to Laserscan 

Run the Gazebo environment:

```console
roslaunch argos_config argos_gazebo.launch ptl:=true
```

On another terminal run:

```console
roslaunch argos_config slam.launch
```

### Depth Camera to Laserscan 

Run the Gazebo environment:

```console
roslaunch argos_config dc_gazebo.launch
```

On another terminal run:

```console
roslaunch argos_config dc_slam.launch
```

### Disparity to Laserscan with RVIZ Robot Position from Ground Truth

Run the Gazebo environment:

```console
roslaunch argos_config argos_gazebo.launch ground_truth:=true
```

On another terminal run:

```console
roslaunch argos_config slam.launch ground_truth:=true
```

### PointCloud to Laserscan with RVIZ Robot Position from Ground Truth

Run the Gazebo environment:

```console
roslaunch argos_config argos_gazebo.launch ground_truth:=true ptl:=true
```

On another terminal run:

```console
roslaunch argos_config slam.launch ground_truth:=true
```



## Movement

To start moving:

* Click '2D Nav Goal'.
* Click and drag at the position you want the robot to go.

## DEM as Ground With ARGOS

Source Gazebo setup.sh

```console
source /usr/share/gazebo/setup.sh
```

Run the Gazebo environment with ARGOS:

```console
roslaunch argos_config vine_argos.launch
```

## DEM as Ground With Husky


### Husky PointCloud to Laserscan with four Cameras
Run the Gazebo environment with Husky:

```console
roslaunch argos_config husky_gazebo.launch ptl:=true
```

### Husky Disparity to Laserscan with four Cameras

Run the Gazebo environment with Husky:

```console
roslaunch argos_config husky_gazebo.launch
```

Run the Gmapping Packages for Husky:

```console
roslaunch argos_config husky_gmapping.launch
```

To launch Husky with 1 ZED camera :

```console
roslaunch argos_config husky_gazebo.launch four_cameras:=false
```