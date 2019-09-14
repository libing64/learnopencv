cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(alpha_blend alphaBlend.cpp)
target_link_libraries(alpha_blend ${OpenCV_LIBS})
