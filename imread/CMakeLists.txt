cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(imread_examples imread_examples.cpp)
target_link_libraries(imread_examples ${OpenCV_LIBS})
