cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(image_alignment image_alignment.cpp)
target_link_libraries(image_alignment ${OpenCV_LIBS})

add_executable(image_alignment_simple_example image_alignment_simple_example.cpp)
target_link_libraries(image_alignment_simple_example ${OpenCV_LIBS})
