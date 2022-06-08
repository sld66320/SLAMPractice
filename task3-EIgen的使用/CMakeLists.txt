cmake_minimum_required(VERSION 2.8)
project(EigienPractice)
find_package(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

add_executable(coordinateTrans coordinateTrans.cpp)
add_executable(useMatrix useMatrix.cpp)
add_executable(solvingLSQ solvingLSQ.cpp)