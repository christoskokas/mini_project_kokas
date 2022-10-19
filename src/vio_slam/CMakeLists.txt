cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 17)
project(vio_slam)



find_package(OpenCV 4.2.0)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.2)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
  endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy 
  std_msgs 
  genmsg 
  sensor_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  nav_msgs
  message_filters
  image_transport
)
find_package(PCL REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

## Declare ROS messages and services
add_message_files(
  FILES
  Num.msg
)

add_service_files(
  FILES
  AddTwoInts.srv
)

add_library(${PROJECT_NAME} SHARED
src/Camera.cpp
src/FeatureDrawer.cpp
src/Frame.cpp
src/trial.cpp
src/Settings.cpp
src/FeatureExtractor.cpp
src/FeatureMatcher.cpp
src/PoseEstimator.cpp
src/FeatureManager.cpp
include/Camera.h
include/FeatureDrawer.h
include/Frame.h
include/trial.h
include/Settings.h
include/FeatureExtractor.h
include/FeatureMatcher.h
include/PoseEstimator.h
include/FeatureManager.h
)

## Generate added messages and services
generate_messages(DEPENDENCIES 
std_msgs
nav_msgs
)

## Declare a catkin package
catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}  
  ${PCL_INCLUDE_DIRS} 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add_executable(camera_subscriber src/camera_subscriber.cpp)
# target_link_libraries(camera_subscriber ${PCL_LIBRARIES} ${catkin_LIBRARIES})

# add_executable(Camera src/Camera.cpp)
# target_link_libraries(Camera ${PCL_LIBRARIES} ${catkin_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
${catkin_LIBRARIES}
${OpenCV_LIBS}
${PCL_LIBRARIES}
${Pangolin_LIBRARIES}
${EIGEN3_LIBS}
${CERES_LIBRARIES}
${YAML_CPP_LIBRARIES}
)

add_executable(VioSlam src/VioSlam.cpp)
target_link_libraries(VioSlam ${PROJECT_NAME})

add_executable(Trial src/trials.cpp)
target_link_libraries(Trial ${PROJECT_NAME})
