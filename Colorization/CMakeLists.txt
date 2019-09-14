cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(colorize_image colorizeImage.cpp)
target_link_libraries(colorize_image ${OpenCV_LIBS})


add_executable(colorize_video colorizeVideo.cpp)
target_link_libraries(colorize_video ${OpenCV_LIBS})