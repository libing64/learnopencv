cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(cmake_example example.cpp)
target_link_libraries(cmake_example ${OpenCV_LIBS})
