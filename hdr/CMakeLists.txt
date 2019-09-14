cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(hdr hdr.cpp)
target_link_libraries(hdr ${OpenCV_LIBS})
