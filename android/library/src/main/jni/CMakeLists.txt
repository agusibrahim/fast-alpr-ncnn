cmake_minimum_required(VERSION 3.10)
project(fast_alpr_ncnn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP compiler flags and disable RTTI/exceptions to match NCNN build config
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fno-rtti -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fno-rtti -fno-exceptions")

# Configure NCNN paths
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn")

# Include paths
include_directories(
    .
    "${NCNN_DIR}/${ANDROID_ABI}/include"
    "${NCNN_DIR}/${ANDROID_ABI}/include/ncnn"
)

# Declare the JNI library
add_library(fast_alpr_ncnn SHARED fast_alpr_jni.cpp)

# Helper for library paths
set(NCNN_LIB_DIR "${NCNN_DIR}/${ANDROID_ABI}/lib")

# Link Android NDK libraries, static NCNN library, Vulkan/glslang dependencies, and OpenMP
target_link_libraries(fast_alpr_ncnn
    "${NCNN_LIB_DIR}/libncnn.a"
    "${NCNN_LIB_DIR}/libglslang.a"
    "${NCNN_LIB_DIR}/libSPIRV.a"
    "${NCNN_LIB_DIR}/libMachineIndependent.a"
    "${NCNN_LIB_DIR}/libOSDependent.a"
    "${NCNN_LIB_DIR}/libGenericCodeGen.a"
    "${NCNN_LIB_DIR}/libOGLCompiler.a"
    vulkan
    omp
    jnigraphics
    android
    log
)
