cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(ocr_simple ocr_simple.cpp)
target_link_libraries(ocr_simple ${OpenCV_LIBS})
