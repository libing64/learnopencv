cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(eigen_face EigenFace.cpp)
target_link_libraries(eigen_face ${OpenCV_LIBS})
