cmake_minimum_required(VERSION 3.10)
project(shallowriver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

# TensorRT paths (Adjust if TensorRT is installed in a custom location)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_LIBRARY nvinfer)
find_library(TENSORRT_PARSER_LIBRARY nvonnxparser)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please check your TensorRT installation.")
endif()

set(SOURCES
    src/main.cpp
    src/bounding_box.cpp
    src/intrinsics.cpp
    src/pose_estimation.cpp
    src/runner.cpp
    src/text_logger.cpp
    src/visualizer.cpp
	src/glogger.cpp
)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PARSER_LIBRARY}
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/res
    ${CMAKE_CURRENT_BINARY_DIR}/res
    COMMENT "Copying /res directory (.onnx, .yaml) to the build directory..."
)
