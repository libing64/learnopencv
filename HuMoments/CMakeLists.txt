cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(HuMoments HuMoments.cpp)
target_link_libraries(HuMoments ${OpenCV_LIBS})

add_executable(shapeMatcher shapeMatcher.cpp)
target_link_libraries(shapeMatcher ${OpenCV_LIBS})
