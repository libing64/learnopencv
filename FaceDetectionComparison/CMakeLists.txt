cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(face_detection_opencv_haar face_detection_opencv_haar.cpp)
target_link_libraries(face_detection_opencv_haar ${OpenCV_LIBS})
