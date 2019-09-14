cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(ssearch ssearch.cpp)
target_link_libraries(ssearch ${OpenCV_LIBS})
