cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(ninjaEyeDetector ninjaEyeDetector.cpp)
target_link_libraries(ninjaEyeDetector ${OpenCV_LIBS})
