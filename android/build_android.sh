#!/bin/bash
set -e

# Change directory to script location
cd "$(dirname "$0")"

echo "============================================="
echo "  Android Standalone Build (Without Gradle)  "
echo "============================================="

# 1. Auto-detect Android SDK & NDK
if [ -d "$HOME/Library/Android/sdk" ]; then
    SDK_DIR="$HOME/Library/Android/sdk"
elif [ -n "$ANDROID_SDK_ROOT" ] && [ -d "$ANDROID_SDK_ROOT" ]; then
    SDK_DIR="$ANDROID_SDK_ROOT"
elif [ -n "$ANDROID_HOME" ] && [ -d "$ANDROID_HOME" ]; then
    SDK_DIR="$ANDROID_HOME"
else
    echo "[-] Android SDK not found. Please set ANDROID_SDK_ROOT or ANDROID_HOME."
    exit 1
fi
echo "[+] Android SDK: $SDK_DIR"

NDK_DIR=$(ls -d ${SDK_DIR}/ndk/* 2>/dev/null | sort -V | tail -n 1)
if [ -z "$NDK_DIR" ]; then
    echo "[-] Android NDK not found under $SDK_DIR/ndk/"
    exit 1
fi
echo "[+] Android NDK: $NDK_DIR"

OMP_SO=$(find "${NDK_DIR}/toolchains/llvm/prebuilt" -path "*/lib/linux/aarch64/libomp.so" | head -n 1)
if [ -z "$OMP_SO" ]; then
    echo "[-] libomp.so not found under $NDK_DIR"
    exit 1
fi
echo "[+] Found libomp.so: $OMP_SO"

BUILD_TOOLS=$(ls -d ${SDK_DIR}/build-tools/* 2>/dev/null | sort -V | tail -n 1)
if [ -z "$BUILD_TOOLS" ]; then
    echo "[-] Android Build Tools not found under $SDK_DIR/build-tools/"
    exit 1
fi
echo "[+] Build Tools: $BUILD_TOOLS"

PLATFORM_JAR=$(ls -d ${SDK_DIR}/platforms/android-*/android.jar 2>/dev/null | sort -V | tail -n 1)
if [ -z "$PLATFORM_JAR" ]; then
    echo "[-] android.jar not found under $SDK_DIR/platforms/"
    exit 1
fi
echo "[+] Platform SDK Jar: $PLATFORM_JAR"

# 2. Download & setup NCNN Android SDK (vulkan release)
NCNN_SDK_URL="https://github.com/Tencent/ncnn/releases/download/20240410/ncnn-20240410-android-vulkan.zip"
NCNN_ZIP_NAME="ncnn-20240410-android-vulkan.zip"
NCNN_DIR="library/src/main/jni/3rdparty/ncnn"

mkdir -p library/src/main/jni/3rdparty

if [ ! -d "$NCNN_DIR" ]; then
    echo "[+] Downloading NCNN Android SDK..."
    curl -L "$NCNN_SDK_URL" -o "$NCNN_ZIP_NAME"
    echo "[+] Extracting NCNN Android SDK..."
    unzip -q "$NCNN_ZIP_NAME"
    
    # Structure it as expected by CMakeLists.txt
    mv ncnn-20240410-android-vulkan "$NCNN_DIR"
    rm -f "$NCNN_ZIP_NAME"
    echo "[+] NCNN SDK set up successfully."
else
    echo "[+] NCNN SDK already exists locally."
fi

# 3. Clean up build directories
echo "[+] Cleaning previous build outputs..."
rm -rf library/build sample/build
mkdir -p library/build sample/build

# 4. Compile C++ Library for arm64-v8a via CMake NDK
echo "[+] Compiling C++ JNI bridge for arm64-v8a..."
mkdir -p library/build/arm64-v8a
cmake \
    -DCMAKE_TOOLCHAIN_FILE="${NDK_DIR}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -S library/src/main/jni \
    -B library/build/arm64-v8a
cmake --build library/build/arm64-v8a -j2

# 5. Compile Java Library source files
echo "[+] Compiling Java Library source files..."
mkdir -p library/build/classes
javac \
    -bootclasspath "$PLATFORM_JAR" \
    -source 1.8 -target 1.8 \
    -d library/build/classes \
    library/src/main/java/com/agusibrahim/fastalpr/*.java

# Pack classes into classes.jar
echo "[+] Packaging classes.jar..."
jar cvf library/build/classes.jar -C library/build/classes .

# 6. Build the AAR (FastAlpr.aar)
echo "[+] Packaging library AAR..."
mkdir -p library/build/aar-dir/jni/arm64-v8a
mkdir -p library/build/aar-dir/assets

cp library/src/main/AndroidManifest.xml library/build/aar-dir/
cp library/build/classes.jar library/build/aar-dir/
cp library/build/arm64-v8a/libfast_alpr_ncnn.so library/build/aar-dir/jni/arm64-v8a/
cp "$OMP_SO" library/build/aar-dir/jni/arm64-v8a/
# Bundle model files directly inside AAR assets
cp ../models/yolo.param ../models/yolo.bin ../models/ocr.param ../models/ocr_patched.bin library/build/aar-dir/assets/

cd library/build/aar-dir
zip -q -r ../FastAlpr.aar . -x "assets/*" "jni/*"
zip -q -r -0 ../FastAlpr.aar assets/ jni/
cd ../../..

echo "[✓] Library AAR ready: library/build/FastAlpr.aar"

# 7. Compile & Build Sample Application APK
echo "[+] Compiling XML resources for sample app..."
mkdir -p sample/build/res-compiled
"${BUILD_TOOLS}/aapt2" compile \
    --dir sample/src/main/res \
    -o sample/build/res-compiled/

echo "[+] Linking XML resources and generating R.java..."
mkdir -p sample/build/gen
# Get all compiled resource .flat files
RES_FILES=$(find sample/build/res-compiled -name "*.flat")
"${BUILD_TOOLS}/aapt2" link \
    -I "$PLATFORM_JAR" \
    --manifest sample/src/main/AndroidManifest.xml \
    --min-sdk-version 24 \
    --target-sdk-version 34 \
    --java sample/build/gen \
    -o sample/build/resources-debug.ap_ \
    ${RES_FILES}

echo "[+] Compiling Java source files for sample app..."
mkdir -p sample/build/classes
javac \
    -bootclasspath "$PLATFORM_JAR" \
    -classpath library/build/classes.jar \
    -source 1.8 -target 1.8 \
    -d sample/build/classes \
    sample/src/main/java/com/agusibrahim/fastalpr/sample/*.java \
    sample/build/gen/com/agusibrahim/fastalpr/sample/R.java

echo "[+] Dexing compiled Java classes (App & Library)..."
mkdir -p sample/build/dex
"${BUILD_TOOLS}/d8" \
    --lib "$PLATFORM_JAR" \
    --output sample/build/dex \
    $(find sample/build/classes -name "*.class") \
    library/build/classes.jar

echo "[+] Building un-signed APK..."
cp sample/build/resources-debug.ap_ sample/build/sample-unsigned.apk

mkdir -p sample/build/apk-content/lib/arm64-v8a
mkdir -p sample/build/apk-content/assets

cp sample/build/dex/classes.dex sample/build/apk-content/
cp library/build/arm64-v8a/libfast_alpr_ncnn.so sample/build/apk-content/lib/arm64-v8a/
cp "$OMP_SO" sample/build/apk-content/lib/arm64-v8a/
# Add model files to assets
cp ../models/yolo.param ../models/yolo.bin ../models/ocr.param ../models/ocr_patched.bin sample/build/apk-content/assets/
# Add sample photos to assets
cp ../65f9903cda758.jpg ../20220507_165417.jpg sample/build/apk-content/assets/

cd sample/build/apk-content
zip -q -r ../sample-unsigned.apk . -x "assets/*" "lib/*"
zip -q -r -0 ../sample-unsigned.apk assets/ lib/
cd ../../..

echo "[+] Aligning APK..."
"${BUILD_TOOLS}/zipalign" -v -f 4 \
    sample/build/sample-unsigned.apk \
    sample/build/sample-aligned.apk > /dev/null

# Sign APK using temporary debug keystore
if [ ! -f debug.keystore ]; then
    echo "[+] Generating temporary debug keystore..."
    keytool -genkeypair -v \
        -keystore debug.keystore \
        -alias androiddebugkey \
        -keypass android \
        -storepass android \
        -keyalg RSA \
        -keysize 2048 \
        -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US"
fi

echo "[+] Signing APK..."
"${BUILD_TOOLS}/apksigner" sign \
    --ks debug.keystore \
    --ks-pass pass:android \
    --key-pass pass:android \
    --out sample/build/sample-debug.apk \
    sample/build/sample-aligned.apk

echo "============================================="
echo "[✓] BUILD SUCCESSFUL!"
echo "    - AAR: android/library/build/FastAlpr.aar"
echo "    - APK: android/sample/build/sample-debug.apk"
echo "============================================="
