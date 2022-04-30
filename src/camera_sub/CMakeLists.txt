cmake_minimum_required(VERSION 3.0.2)
project(camera_sub)



## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)
find_package(PCL REQUIRED)

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
include_directories(${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(camera_subscriber src/camera_subscriber.cpp)
target_link_libraries(camera_subscriber ${PCL_LIBRARIES} ${catkin_LIBRARIES})
