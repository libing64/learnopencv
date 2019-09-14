cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(blob_detector blob.cpp)
target_link_libraries(blob_detector ${OpenCV_LIBS})
