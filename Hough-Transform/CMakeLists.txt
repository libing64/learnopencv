cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(hough_lines hough_lines.cpp)
target_link_libraries(hough_lines ${OpenCV_LIBS})


add_executable(hough_circles hough_circles.cpp)
target_link_libraries(hough_circles ${OpenCV_LIBS})
