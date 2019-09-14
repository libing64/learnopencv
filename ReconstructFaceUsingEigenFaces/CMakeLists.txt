cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(reconstructFace reconstructFace.cpp)
target_link_libraries(reconstructFace ${OpenCV_LIBS})

add_executable(createPCAModel createPCAModel.cpp)
target_link_libraries(createPCAModel ${OpenCV_LIBS})
