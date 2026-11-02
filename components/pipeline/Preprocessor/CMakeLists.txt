cmake_minimum_required(VERSION 3.16)
project(exam_hacker_preprocessor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    photo
    cudaarithm
    cudaimgproc
    cudawarping
)

add_library(exam_preprocessor
    src/Preprocessor.cpp
)
target_include_directories(exam_preprocessor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(exam_preprocessor PUBLIC ${OpenCV_LIBS})

add_executable(exam_preprocess src/main.cpp)
target_link_libraries(exam_preprocess PRIVATE exam_preprocessor)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV CUDA modules: ${OpenCV_cudaarithm_FOUND}; ${OpenCV_cudaimgproc_FOUND}; ${OpenCV_cudawarping_FOUND}")
