cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(imfill imfill.cpp)
target_link_libraries(imfill ${OpenCV_LIBS})
