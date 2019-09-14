cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(goturnTracker goturnTracker.cpp)
target_link_libraries(goturnTracker ${OpenCV_LIBS})
