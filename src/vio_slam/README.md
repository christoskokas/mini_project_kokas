# MC-VSLAM
This is the software related to the publication: Multicamera Visual SLAM For Vineyard Inspection, submited at the CASE 2023.

MC-VSLAM is a multicamera visual SLAM designed for vineyard inspection. To address the challenge of homogeneous environments, loop closures are detected using AprilTags.

MC-VSLAM has been tested with OpenCV 4.2.0, Eigen 3.3.7 on Ubuntu 20.04 with ROS noetic.

## Installation

### Prerequisites

Install Eigen :

```
sudo apt update
sudo apt install libeigen3-dev
```

Install OpenCV :

```
sudo apt install libopencv-dev python3-opencv
```

### Pangolin

MC-VSLAM uses Pangolin for visualization. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### MC-VSLAM Installation

Clone the package on the catkin workspace.

```
git clone https://github.com/christoskokas/mc_vslam.git
```

Install Ceres-Solver from this [link](http://ceres-solver.org/installation.html). The files are provided in the folder Thirdparty inside this package.

From the catkin workspace install all the needed dependencies for the packages.

```
rosdep install --from-paths src --ignore-src -r -y
```

Build the workspace.


```
catkin build mc_vslam
source devel/setup.bash
```

### Alternative Way

Build both mc_vslam and ceres solver without ROS ( Real time cannot operate without ROS ) :

```
chmod +x build.sh
./build.sh
```

## Quick Start

Several launch files are provided. The RT denotes real-time and the AT denotes the use of AprilTag Loop Closure. Change the launch files to match the config file name and the topic of the image msgs for AprilTag detection.


### Without ROS

If the installation was completed without ROS, images need to be provided as presented below ( the bullets are folders ): 

- mc_vslam
  - images
    - dataset_name
      - left
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...
      - right
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...
      - leftBack
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...
      - rightBack
        1. 000000.jpg(.png)
        2. 000001.jpg(.png)
        3. ...

And the full path to the dataset folder has to be provided in the config file.

Single Cam is also provided to test on known datasets like KITTI or EUROC. Configs for both KITTI and EUROC are provided. To run single cam :

```
./SingleCam config_file_name
```

with the appropriate changes on the config file.





