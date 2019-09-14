cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(interactiveColorDetect interactiveColorDetect.cpp)
target_link_libraries(interactiveColorDetect ${OpenCV_LIBS})


add_executable(interactiveColorSegment interactiveColorSegment.cpp)
target_link_libraries(interactiveColorSegment ${OpenCV_LIBS})
