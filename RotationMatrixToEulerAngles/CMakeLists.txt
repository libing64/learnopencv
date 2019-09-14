cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(rotm2euler rotm2euler.cpp)
target_link_libraries(rotm2euler ${OpenCV_LIBS})
