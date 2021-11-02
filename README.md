# mini_project_EMP

Goal of this project is the modelling of the argos quadruped robot using champ setup assistant.

## Ιnstallation

Clone the packages needed on catkin workspace in the src folder or on a git folder and make a symlink to the catkin workspace. 

```console
git clone --recursive https://github.com/christoskokas/mini_project_EMP.git
```

From the catkin workspace install all the needed dependencies for the packages.

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
