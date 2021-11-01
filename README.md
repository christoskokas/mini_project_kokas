# mini_project_EMP

Goal of this project is the modelling of the argos quadruped robot using champ setup assistant.

## Ιnstallation

Clone the packages needed on catkin workspace or on a git folder and make a symlink to the catkin workspace. From this point on, the catkin workspace that will be used is catkin_ws. 

```console
git clone https://github.com/christoskokas/mini_project_EMP.git
git clone --recursive https://github.com/chvmp/champ
git clone https://github.com/chvmp/yocs_velocity_smoother.git
git clone --recursive https://github.com/ros-perception/image_pipeline
git clone https://github.com/skohlbr/disparity_image_proc.git
git clone https://github.com/ros-perception/depthimage_to_laserscan.git
git clone --branch release/0.62-noetic \https://github.com/stonier/ecl_core.git
```

From catkin_ws install all the needed dependencies for the packages.

```console
rosdep install --from-paths src --ignore-src -r -y
```

Build the workspace.

```console
catkin build argos_config
source devel/setup.bash
```

## Quick Start

Run the Gazebo environment:

```console
roslaunch argos_config gazebo.launch
```

On another terminal run:

```console
roslaunch argos_config slam.launch
```

To start moving:

* Click '2D Nav Goal'.
* Click and drag at the position you want the robot to go.
