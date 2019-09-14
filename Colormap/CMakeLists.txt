cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(colormap colormap.cpp)
target_link_libraries(colormap ${OpenCV_LIBS})

add_executable(custom_colormap custom_colormap.cpp)
target_link_libraries(custom_colormap ${OpenCV_LIBS})
