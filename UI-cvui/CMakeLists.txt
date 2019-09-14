cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(hello-world hello-world.cpp)
target_link_libraries(hello-world ${OpenCV_LIBS})


add_executable(canny canny.cpp)
target_link_libraries(canny ${OpenCV_LIBS})
