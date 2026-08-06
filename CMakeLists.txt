cmake_minimum_required(VERSION 3.10)
project(shallowriver VERSION 0.1.0 LANGUAGES CXX)

# Default to /usr instead of CMake's usual /usr/local so a plain `make install`
# lands in the same place `cpack` packages it (see CPACK_PACKAGING_INSTALL_PREFIX
# below) -- both need to agree since it's baked into the binary as
# SHALLOWRIVER_INSTALL_ASSET_DIR further down.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Install path prefix" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

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
	src/paths.cpp
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

# Fallback location for the read-only .onnx model assets when no local ./res
# directory is found next to the running binary (see src/paths.cpp) -- i.e.
# once installed via `make install`/apt rather than run from the build tree.
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHALLOWRIVER_INSTALL_ASSET_DIR="${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/res"
)

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

# --- Install rules (required for `cpack`/apt packaging) ---
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/res/
    DESTINATION share/${PROJECT_NAME}/res
)

# --- CPack / .deb packaging ---
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT "comicallynormal@gmail.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Multi-camera body-pose pipeline optimized for latency on Jetson Orin")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "comicallynormal@gmail.com")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "nvidia-jetpack, libeigen3-dev")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

include(CPack)
