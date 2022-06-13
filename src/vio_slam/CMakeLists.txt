cmake_minimum_required(VERSION 3.0.2)
project(vio_slam)



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

add_library(${PROJECT_NAME} SHARED
src/Camera.cpp
src/VioSlam.cpp
include/Camera.h
)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}  
  ${PCL_INCLUDE_DIRS} 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add_executable(camera_subscriber src/camera_subscriber.cpp)
# target_link_libraries(camera_subscriber ${PCL_LIBRARIES} ${catkin_LIBRARIES})

# add_executable(Camera src/Camera.cpp)
# target_link_libraries(Camera ${PCL_LIBRARIES} ${catkin_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
${catkin_LIBRARIES}
${PCL_LIBRARIES}
)

add_executable(VioSlam src/VioSlam.cpp)
target_link_libraries(VioSlam ${PROJECT_NAME})
