cmake_minimum_required(VERSION 3.1)


find_package(OpenCV REQUIRED)

add_executable(barcode_qrcode_scanner barcode-QRcodeScanner.cpp)
target_link_libraries(barcode_qrcode_scanner ${OpenCV_LIBS})
