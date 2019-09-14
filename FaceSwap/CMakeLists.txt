cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(faceSwap faceSwap.cpp)
target_link_libraries(faceSwap ${OpenCV_LIBS})
