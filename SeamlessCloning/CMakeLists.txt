cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(clone clone.cpp)
target_link_libraries(clone ${OpenCV_LIBS})
