cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(frame_rate frame_rate.cpp)
target_link_libraries(frame_rate ${OpenCV_LIBS})
