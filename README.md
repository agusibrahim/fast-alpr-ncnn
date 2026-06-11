# Ultra-Fast C++ ALPR & OCR (NCNN-Powered)

An extremely fast, lightweight, and self-contained Automatic License Plate Recognition (ALPR) engine written in C++. Leveraging **YOLOv9** (416px resolution) for detection and **CCT-XS** (CRNN/CTC) for OCR, this project achieves near-instantaneous plate reading on standard CPUs.

### 🌟 Key Features

* **Blazing Fast**: Processes multiple plates sequentially in **~149 ms** on standard CPUs (~75 ms per plate).
* **Zero Dependencies**: Builds into a clean, portable package with models included. No heavy machine learning frameworks required at runtime.
* **Cross-Platform**: Native Desktop CLI (macOS, Linux, Windows), Android JNI Library (AAR), and WebAssembly (WASM) for web browser deployments.
* **Smart Compilation**: Automated build setup. Apple Silicon uses a fast precompiled library, while Linux and Windows compile NCNN from source automatically.
* **Accurate OCR Decoding**: Decodes multi-head CCT-XS output without CTC duplicate collapsing, accurately recognizing consecutive matching characters (e.g., `1111` or `MM`).
* **Edge-Aware Padding**: Bounding boxes are automatically padded (8% width expand left & right) to guarantee that characters at the plate boundaries (e.g., `D` or `M`) are not cut off during cropping.

---

## 🚀 Speed Benchmarks

The following benchmarks were conducted on an Apple Silicon (M-series) host with a batch iteration of 5 runs on a test image containing **two license plates** (`65f9903cda758.jpg`):

| Engine | Avg Latency (ms) | Speedup vs Python | Runtime Dependencies | Self-Contained |
|:---|:---:|:---:|:---|:---:|
| 🏎️ **NCNN C++** | **149 ms** | **3.6x** | None (Static CPU, OpenMP) | **Yes (models included)** |
| 🦀 **Rust** | **268 ms** | **2.0x** | ONNX Runtime lib | No |
| 🐍 **Python** | **531 ms** | **1.0x (Baseline)** | Python interpreter, ORT, OpenCV | No |

---

## 📸 Sample Visualizations

Detected license plate bounding boxes and OCR results drawn on the input image:

### Input Traffic Scene (`65f9903cda758.jpg`)
![Input Image](./65f9903cda758.jpg)

### Detected Output Bounding Boxes (`result.jpg`)
![Output Visualization](./result.jpg)

---

## 🛠️ Build Prerequisites

### macOS
* **Xcode Command Line Tools**
* **CMake** (`brew install cmake`)
* Apple Silicon macOS (ARM64) uses the precompiled static library `3rdparty/ncnn/lib/libncnn.a` for instant compilation.
* Intel macOS (x86_64) automatically falls back to CMake `FetchContent` to compile NCNN from source.

### Linux
* **CMake** (version >= 3.14) and **Build Essentials** (gcc/g++)
* **Git** (required for the source-compilation fallback)
* *Note: If NCNN is not found globally on your system, CMake will automatically download and compile NCNN from source. No manual configuration is required.*

---

## 💻 How to Build & Run

From the project root directory:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run the compiled binary:
./alpr_ncnn ../65f9903cda758.jpg
```

---

## 📦 Releases & Prebuilt Binaries

Prebuilt binaries for all supported platforms (macOS, Linux, Windows, Android AAR, and WebAssembly) are automatically built and packaged.

You can download the latest standalone binaries, Android libraries, and WebAssembly packages from the [GitHub Releases](https://github.com/agusibrahim/fast-alpr-ncnn/releases) page.
