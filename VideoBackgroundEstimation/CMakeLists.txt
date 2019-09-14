cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(removeVideoBg removeVideoBg.cpp)
target_link_libraries(removeVideoBg ${OpenCV_LIBS})
