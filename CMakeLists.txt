cmake_minimum_required(VERSION 3.14)
project(alpr_ncnn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
if(NOT MSVC)
    add_compile_options(-O3 -march=native)
endif()

# Define local standalone directories
set(NCNN_3RDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn")

# Include directories configured below target-locally

# Executable
add_executable(alpr_ncnn main.cpp)
target_include_directories(alpr_ncnn PRIVATE
    .
    "${NCNN_3RDPARTY_DIR}/include"
)

# Link NCNN static library (smart check for system package on Linux vs local fallback on macOS)
if(NOT APPLE)
    find_package(ncnn QUIET)
endif()

if(ncnn_FOUND)
    message(STATUS "Found NCNN system package. Linking system ncnn...")
    target_link_libraries(alpr_ncnn PRIVATE ncnn)
elseif(APPLE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64"))
    message(STATUS "Linking local macOS static library (arm64)...")
    target_link_libraries(alpr_ncnn PRIVATE "${NCNN_3RDPARTY_DIR}/lib/libncnn.a")
else()
    message(STATUS "NCNN not found on system. Fetching and compiling NCNN from source...")
    include(FetchContent)
    
    # Configure NCNN build options to only build the CPU library (Release mode)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(NCNN_DISABLE_RTTI OFF CACHE BOOL "" FORCE)
    set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn.git
        GIT_TAG 20240410
        GIT_SUBMODULES ""
    )
    FetchContent_MakeAvailable(ncnn)
    target_link_libraries(alpr_ncnn PRIVATE ncnn)
endif()

# Link threads library
find_package(Threads REQUIRED)
target_link_libraries(alpr_ncnn PRIVATE Threads::Threads)

# On macOS, NCNN links Accelerate framework for hardware speedups if built with it
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(alpr_ncnn PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endif()

# Post-build copy models directory to target directory
add_custom_command(TARGET alpr_ncnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/models"
    "$<TARGET_FILE_DIR:alpr_ncnn>/models"
)
