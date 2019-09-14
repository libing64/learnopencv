cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(textDetection textDetection.cpp)
target_link_libraries(textDetection ${OpenCV_LIBS})
