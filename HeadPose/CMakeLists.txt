cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(headPose headPose.cpp)
target_link_libraries(headPose ${OpenCV_LIBS})
