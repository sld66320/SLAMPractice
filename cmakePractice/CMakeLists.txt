cmake_minimum_required(VERSION 3.10)
project(cmakePractice)

set(CMAKE_CXX_STANDARD 14)
add_library(hellolib libHello.cpp)
add_library(hellolib_shared SHARED libHello.cpp)
add_executable(main main.cpp)
target_link_libraries(main hellolib_shared)
