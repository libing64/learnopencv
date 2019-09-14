cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(removeRedEyes removeRedEyes.cpp)
target_link_libraries(removeRedEyes ${OpenCV_LIBS})
