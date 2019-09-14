cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(qrCodeOpencv qrCodeOpencv.cpp)
target_link_libraries(qrCodeOpencv ${OpenCV_LIBS})
