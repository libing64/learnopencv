cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(multiTracker multiTracker.cpp)
target_link_libraries(multiTracker ${OpenCV_LIBS})
