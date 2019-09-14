cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(remove_red_eye removeRedEyes.cpp)
target_link_libraries(remove_red_eye ${OpenCV_LIBS})
