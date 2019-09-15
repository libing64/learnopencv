cmake_minimum_required(VERSION 3.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(stereo_calibrate stereo_calibrate.cpp)
target_link_libraries(stereo_calibrate ${OpenCV_LIBS} "-lpopt")


add_executable(mono_calibrate mono_calibrate.cpp)
target_link_libraries(mono_calibrate ${OpenCV_LIBS} "-lpopt")