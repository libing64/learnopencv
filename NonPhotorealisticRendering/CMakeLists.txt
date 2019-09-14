cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(npr npr.cpp)
target_link_libraries(npr ${OpenCV_LIBS})
