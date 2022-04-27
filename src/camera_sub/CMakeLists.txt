cmake_minimum_required(VERSION 3.0.2)
project(camera_sub)



## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

## Declare ROS messages and services
add_message_files(
  FILES
  Num.msg
)

add_service_files(
  FILES
  AddTwoInts.srv
)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})

add_executable(camera_subscriber src/camera_subscriber.cpp)
target_link_libraries(camera_subscriber ${catkin_LIBRARIES})
