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


### Husky movement with keyboard for VO

Run The Gazebo environment with Husky:

```console
roslaunch argos_config husky_gazebo.launch
```

Run the teleop package with the cmd_vel topic that controls the Husky's velocity:

```console
rosrun key_teleop key_teleop.py key_vel:=cmd_vel
```

(Optional) Run the Rviz environment:

```console
roslaunch argos_config husky_gmapping.launch
```

## ORB_SLAM3 for ROS

### Prerequisites

ORB_SLAM3_noetic has been tested on Ubuntu 20.04 with ROS Noetic, Eigen 3.3.7, OpenCV 4.2.0 and Pangolin 0.8.

Install Eigen using the command:

```console
sudo apt update
sudo apt install libeigen3-dev
```

Install OpenCV:

```console
sudo apt install libopencv-dev python3-opencv
```

### Pangolin

ORB_SLAM3 uses Pangolin for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### Installing ORB_SLAM3_noetic

* Download ORB_SLAM3_noetic package:

```console
git clone --recursive https://github.com/christoskokas/ORB_SLAM3_noetic.git
```

* Execute build.sh script to build thirdparty packages:

```console
chmod +x build.sh
./build.sh
```

* Build the ROS package:

```console
catkin build orbslam3
```

### ORB_SLAM3 With Husky

Run The Gazebo environment with Husky:

```console
roslaunch argos_config husky_gazebo.launch
```

Run the teleop package with the cmd_vel topic that controls the Husky's velocity:

```console
rosrun key_teleop key_teleop.py key_vel:=cmd_vel
```

#### Stereo ORB_SLAM3

Run the ORB_SLAM3 package with stereo:

```console
roslaunch orbslam3 ros_stereo.launch 
```

#### Stereo-Inertial ORB_SLAM3

Run the ORB_SLAM3 package with stereo inertial:

```console
roslaunch orbslam3 ros_stereo_inertial.launch 
```

