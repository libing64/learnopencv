cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(inpaint inpaint.cpp)
target_link_libraries(inpaint ${OpenCV_LIBS})
