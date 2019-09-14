cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(homography homography.cpp)
target_link_libraries(homography ${OpenCV_LIBS})

add_executable(homography2 homography2.cpp)
target_link_libraries(homography2 ${OpenCV_LIBS})