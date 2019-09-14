cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(Invisibility_Cloak Invisibility_Cloak.cpp)
target_link_libraries(Invisibility_Cloak ${OpenCV_LIBS})
