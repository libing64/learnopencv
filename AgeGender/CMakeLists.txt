cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(age_gender AgeGender.cpp)
target_link_libraries(age_gender ${OpenCV_LIBS})
