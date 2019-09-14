cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(convex_hull example.cpp)
target_link_libraries(convex_hull ${OpenCV_LIBS})
