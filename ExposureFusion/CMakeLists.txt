cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(exposureFusion exposureFusion.cpp)
target_link_libraries(exposureFusion ${OpenCV_LIBS})
