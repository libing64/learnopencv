cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(digits_classification train_digits.cpp)
target_link_libraries(digits_classification ${OpenCV_LIBS})
