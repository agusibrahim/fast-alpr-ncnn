#!/bin/bash
set -e

# Change directory to script location
cd "$(dirname "$0")"

echo "============================================="
# 1. Setup Emscripten SDK
if command -v emcc >/dev/null 2>&1; then
    echo "[+] emcc is already available in PATH. Skipping local emsdk setup."
else
    if [ ! -d "emsdk" ]; then
        echo "[+] Cloning Emscripten SDK..."
        git clone --depth 1 https://github.com/emscripten-core/emsdk.git
        cd emsdk
        echo "[+] Installing Emscripten compiler..."
        ./emsdk install latest
        echo "[+] Activating Emscripten compiler..."
        ./emsdk activate latest
        cd ..
    fi
    # Activate emsdk environment
    source emsdk/emsdk_env.sh
fi

# 2. Clone and compile Tencent NCNN from source targeting WebAssembly
if [ ! -d "ncnn" ]; then
    echo "[+] Cloning Tencent NCNN..."
    git clone --depth 1 -b 20240410 https://github.com/Tencent/ncnn.git
fi

if [ ! -d "ncnn/build-wasm" ]; then
    echo "[+] Compiling NCNN for WebAssembly..."
    mkdir -p ncnn/build-wasm
    cd ncnn/build-wasm
    emcmake cmake \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DNCNN_BUILD_TESTS=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TOOLS=OFF \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_SIMPLEOCV=ON \
        ..
    emmake make -j2
    cd ../..
else
    echo "[+] NCNN WASM build already exists."
fi

# 3. Compile wrapper fast_alpr_wasm.js/wasm
echo "[+] Compiling WebAssembly bindings..."
rm -rf build
mkdir -p build
cd build
emcmake cmake ..
emmake make -j2
cd ..

# Copy compiled files to root directory
cp build/fast_alpr_wasm.js build/fast_alpr_wasm.wasm .

# 4. Copy models and test images for local server serving
echo "[+] Bundling models and test assets..."
mkdir -p models
cp ../models/yolo.param ../models/yolo.bin ../models/ocr.param ../models/ocr_patched.bin models/
cp ../65f9903cda758.jpg ../20220507_165417.jpg .

echo "============================================="
echo "[✓] WASM BUILD SUCCESSFUL!"
echo "    Files ready:"
echo "    - wasm/fast_alpr_wasm.js"
echo "    - wasm/fast_alpr_wasm.wasm"
echo "============================================="
