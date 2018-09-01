cmake_minimum_required(VERSION 2.8)

project(colorLines)

find_package(OpenCV REQUIRED)
include_directories(include)

# file(GLOB SOURCES "src/*.cpp")

set(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")

add_library(helpers src/colorLines.cpp
					src/graph.cpp
					src/maxflow.cpp)

add_executable(main src/main.cpp)

add_executable(binary src/binary.cpp)

target_link_libraries(main ${OpenCV_LIBS} helpers)

target_link_libraries(binary ${OpenCV_LIBS} helpers)
