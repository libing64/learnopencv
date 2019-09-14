cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(warpTriangle warpTriangle.cpp)
target_link_libraries(warpTriangle ${OpenCV_LIBS})
