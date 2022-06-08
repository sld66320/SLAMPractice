cmake_minimum_required(VERSION 2.8)
project(useOpenCV)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

Find_Package(Ceres REQUIRED)

add_executable(useOpenCV useOpenCV.cpp)
target_link_libraries(useOpenCV ${OpenCV_LIBRARIES} )

add_executable(undistort undistort.cpp)
target_link_libraries(undistort ${OpenCV_LIBRARIES} )
target_link_libraries(undistort ${CERES_LIBRARIES})
