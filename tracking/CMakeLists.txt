cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(tracker tracker.cpp)
target_link_libraries(tracker ${OpenCV_LIBS})
