cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(videoRead videoRead.cpp)
target_link_libraries(videoRead ${OpenCV_LIBS})


add_executable(videoWrite videoWrite.cpp)
target_link_libraries(videoWrite ${OpenCV_LIBS})
