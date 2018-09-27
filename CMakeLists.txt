cmake_minimum_required(VERSION 3.5)
project(libMarkDemo)

set(CMAKE_CXX_STANDARD 11)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/libMark/include)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/libMark)

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(example ${DIR_SRCS})

target_link_libraries(example Mark)

find_package( OpenCV REQUIRED)
target_link_libraries(example  ${OpenCV_LIBS})











