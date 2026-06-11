cmake_minimum_required(VERSION 3.10)
project(fast_alpr_wasm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure Emscripten compile/link options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 --bind -s EXPORTED_RUNTIME_METHODS='[\"FS\", \"HEAPU8\"]' -s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]'")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s WASM=1 -s ALLOW_MEMORY_GROWTH=1")

# Configure NCNN paths
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn")
set(NCNN_BUILD_DIR "${NCNN_DIR}/build-wasm")

# Include paths
include_directories(
    .
    "${NCNN_DIR}/src"
    "${NCNN_BUILD_DIR}/src"
)

# Declare WebAssembly library output
add_executable(fast_alpr_wasm fast_alpr_wasm.cpp)

# Link static NCNN library
target_link_libraries(fast_alpr_wasm
    "${NCNN_BUILD_DIR}/src/libncnn.a"
)

# Set output name and suffix
set_target_properties(fast_alpr_wasm PROPERTIES
    OUTPUT_NAME "fast_alpr_wasm"
    SUFFIX ".js"
)
