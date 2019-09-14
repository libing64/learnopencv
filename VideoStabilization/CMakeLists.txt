cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(video_stabilization video_stabilization.cpp)
target_link_libraries(video_stabilization ${OpenCV_LIBS})
