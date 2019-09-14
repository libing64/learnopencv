cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(forEach forEach.cpp)
target_link_libraries(forEach ${OpenCV_LIBS})
