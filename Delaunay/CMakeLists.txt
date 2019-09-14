cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(delaunay delaunay.cpp)
target_link_libraries(delaunay ${OpenCV_LIBS})
