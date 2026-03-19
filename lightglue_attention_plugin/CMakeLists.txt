cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

project(lightglue_attention_plugin LANGUAGES CXX CUDA)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LIGHTGLUE_SYSTEM_PROCESSOR)
set(LIGHTGLUE_DEFAULT_CUDA_ARCHITECTURES 86)
if(LIGHTGLUE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(LIGHTGLUE_DEFAULT_CUDA_ARCHITECTURES 87)
endif()

if(LIGHTGLUE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
        set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Jetson target detected; defaulting CMAKE_CUDA_ARCHITECTURES to 87 for Orin NX")
    elseif(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "87")
        message(WARNING "Detected aarch64/Jetson but CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}. For Orin NX use -DCMAKE_CUDA_ARCHITECTURES=87, or remove the build directory and reconfigure.")
    endif()
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES ${LIGHTGLUE_DEFAULT_CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -w --use_fast_math")

find_package(CUDAToolkit REQUIRED)

get_filename_component(CUTLASS_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass-v3.4.1/include" ABSOLUTE)

set(TENSORRT_LIB_HINTS
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/lib/aarch-linux-gnu
    /usr/local/lib
)

set(TENSORRT_INCLUDE_HINTS
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu
    /usr/include
    /usr/local/cuda/include
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_INCLUDE_HINTS})
find_library(TENSORRT_NVINFER_LIB nvinfer HINTS ${TENSORRT_LIB_HINTS})
find_library(TENSORRT_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIB_HINTS})

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVINFER_PLUGIN_LIB)
    message(FATAL_ERROR "TensorRT headers or libraries not found. Set TENSORRT_INCLUDE_DIR/TENSORRT_NVINFER_LIB/TENSORRT_NVINFER_PLUGIN_LIB if needed.")
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR      : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES    : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS    : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUTLASS_DIRS                : ${CUTLASS_DIRS}")
message(STATUS "TENSORRT_INCLUDE_DIR        : ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TENSORRT_NVINFER_LIB        : ${TENSORRT_NVINFER_LIB}")
message(STATUS "TENSORRT_NVINFER_PLUGIN_LIB : ${TENSORRT_NVINFER_PLUGIN_LIB}")

add_library(
    lightglue_attention_plugin
    SHARED
    attention_headdim_64_fp16in_fp16out.cu
    attention_headdim_64_fp16in_fp32out.cu
    lightglue_attention_plugin.cpp
    common/checkMacrosPlugin.cpp
)

target_include_directories(
    lightglue_attention_plugin
    PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUTLASS_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    lightglue_attention_plugin
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVINFER_PLUGIN_LIB}
)

set_target_properties(lightglue_attention_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
