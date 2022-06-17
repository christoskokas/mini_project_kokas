cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 14)
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
  cv_bridge
  image_transport
)
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
src/VioSlam.cpp
src/Camera.cpp
src/Viewer.cpp
include/Camera.h
include/Viewer.h
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
${OpenCV_LIBS}
${PCL_LIBRARIES}
)

add_executable(VioSlam src/VioSlam.cpp)
target_link_libraries(VioSlam ${PROJECT_NAME})
