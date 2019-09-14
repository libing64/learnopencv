cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(center_of_multipe_blob center_of_multipe_blob.cpp)
target_link_libraries(center_of_multipe_blob ${OpenCV_LIBS})


add_executable(single_blob single_blob.cpp)
target_link_libraries(single_blob ${OpenCV_LIBS})
