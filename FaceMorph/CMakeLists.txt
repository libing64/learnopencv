cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(faceMorph faceMorph.cpp)
target_link_libraries(faceMorph ${OpenCV_LIBS})
