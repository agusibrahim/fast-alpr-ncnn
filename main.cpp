#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <mutex>
#include <cstdlib>
#include <cstdio>

#include "httplib.h"

// Include NCNN headers
#include "net.h"
#include "mat.h"
#include "cpu.h"
#include "allocator.h"

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread/qos.h>
#endif

// Include stb image headers with ARM NEON SIMD enabled when available
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#ifndef STBI_NEON
#define STBI_NEON
#endif
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Charset for license plate OCR – must match cct_xs_v2_global_plate_config.yaml (37 classes)
const std::vector<char> CHARSET = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '_'
};
const int BLANK_INDEX = CHARSET.size() - 1; // 36 = '_'
bool g_verbose = false;

// Structures
struct BoundingBox {
    int x1, y1, x2, y2;
    float score;
    
    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
    int area() const { return width() * height(); }
};

struct Detection {
    BoundingBox bbox;
    std::string text;
    float ocr_conf;
};

void save_mat(const ncnn::Mat& m, const std::string& path) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp) {
        fwrite(m.data, 1, m.total() * m.elemsize, fp);
        fclose(fp);
        std::cout << "[DEBUG] Saved " << path << " size=" << m.total() * m.elemsize << " bytes" << std::endl;
    }
}

// Single-pass fused letterbox + bilinear resize + /255.0f normalization directly into planar ncnn::Mat(416, 416, 3)
static void letterbox_to_yolo_mat(const unsigned char* src, int src_w, int src_h,
                                  ncnn::Mat& yolo_in, int target_w, int target_h,
                                  float& scale, int& pad_x, int& pad_y,
                                  ncnn::Allocator* allocator = 0) {
    scale = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = (int)std::round(src_w * scale);
    int new_h = (int)std::round(src_h * scale);

    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;

    yolo_in.create(target_w, target_h, 3, sizeof(float), allocator);
    yolo_in.fill(114.0f / 255.0f);

    std::vector<int> x0_tab(new_w), x1_tab(new_w);
    std::vector<float> dx_tab(new_w);
    const float scale_x = (float)src_w / (float)new_w;
    const float scale_y = (float)src_h / (float)new_h;
    for (int x = 0; x < new_w; ++x) {
        float sx = (float)x * scale_x;
        int x0 = (int)sx;
        x0_tab[x] = x0 * 3;
        x1_tab[x] = std::min(x0 + 1, src_w - 1) * 3;
        dx_tab[x] = sx - (float)x0;
    }

    float* ch0 = yolo_in.channel(0);
    float* ch1 = yolo_in.channel(1);
    float* ch2 = yolo_in.channel(2);
    const float inv255 = 1.0f / 255.0f;

    for (int y = 0; y < new_h; ++y) {
        float sy = (float)y * scale_y;
        int y0 = (int)sy;
        int y1 = std::min(y0 + 1, src_h - 1);
        float dy = sy - (float)y0;
        float wy0 = 1.0f - dy;
        float wy1 = dy;

        const unsigned char* row0 = src + y0 * src_w * 3;
        const unsigned char* row1 = src + y1 * src_w * 3;
        int dst_offset = (y + pad_y) * target_w + pad_x;
        float* d0 = ch0 + dst_offset;
        float* d1 = ch1 + dst_offset;
        float* d2 = ch2 + dst_offset;

        for (int x = 0; x < new_w; ++x) {
            int i0 = x0_tab[x];
            int i1 = x1_tab[x];
            float dx = dx_tab[x];
            float w00 = (1.0f - dx) * wy0;
            float w10 = dx * wy0;
            float w01 = (1.0f - dx) * wy1;
            float w11 = dx * wy1;

            float r = w00 * row0[i0 + 0] + w10 * row0[i1 + 0] + w01 * row1[i0 + 0] + w11 * row1[i1 + 0];
            float g = w00 * row0[i0 + 1] + w10 * row0[i1 + 1] + w01 * row1[i0 + 1] + w11 * row1[i1 + 1];
            float b = w00 * row0[i0 + 2] + w10 * row0[i1 + 2] + w01 * row1[i0 + 2] + w11 * row1[i1 + 2];

            d0[x] = (float)((int)r) * inv255;
            d1[x] = (float)((int)g) * inv255;
            d2[x] = (float)((int)b) * inv255;
        }
    }
}

// Single-pass fused ROI crop + bilinear resize + /255.0f normalization directly into planar ncnn::Mat(128, 64, 3)
static void crop_resize_to_ocr_mat(const unsigned char* src, int src_w, int src_h,
                                   int x1, int y1, int x2, int y2,
                                   ncnn::Mat& ocr_in,
                                   ncnn::Allocator* allocator = 0) {
    const int dst_w = 128;
    const int dst_h = 64;
    int crop_w = std::max(1, x2 - x1);
    int crop_h = std::max(1, y2 - y1);

    ocr_in.create(dst_w, dst_h, 3, sizeof(float), allocator);
    float* ch0 = ocr_in.channel(0);
    float* ch1 = ocr_in.channel(1);
    float* ch2 = ocr_in.channel(2);
    const float inv255 = 1.0f / 255.0f;

    int x0_tab[128], x1_tab[128];
    float dx_tab[128];
    const float scale_x = (float)crop_w / (float)dst_w;
    const float scale_y = (float)crop_h / (float)dst_h;
    for (int x = 0; x < dst_w; ++x) {
        float sx = (float)x * scale_x;
        int x0 = (int)sx;
        int x1c = std::min(x0 + 1, crop_w - 1);
        x0_tab[x] = (x1 + x0) * 3;
        x1_tab[x] = (x1 + x1c) * 3;
        dx_tab[x] = sx - (float)x0;
    }

    for (int y = 0; y < dst_h; ++y) {
        float sy = (float)y * scale_y;
        int y0 = (int)sy;
        int y1c = std::min(y0 + 1, crop_h - 1);
        float dy = sy - (float)y0;
        float wy0 = 1.0f - dy;
        float wy1 = dy;

        const unsigned char* row0 = src + (y1 + y0) * src_w * 3;
        const unsigned char* row1 = src + (y1 + y1c) * src_w * 3;
        float* d0 = ch0 + y * dst_w;
        float* d1 = ch1 + y * dst_w;
        float* d2 = ch2 + y * dst_w;

        for (int x = 0; x < dst_w; ++x) {
            int i0 = x0_tab[x];
            int i1 = x1_tab[x];
            float dx = dx_tab[x];
            float w00 = (1.0f - dx) * wy0;
            float w10 = dx * wy0;
            float w01 = (1.0f - dx) * wy1;
            float w11 = dx * wy1;

            float r = w00 * row0[i0 + 0] + w10 * row0[i1 + 0] + w01 * row1[i0 + 0] + w11 * row1[i1 + 0];
            float g = w00 * row0[i0 + 1] + w10 * row0[i1 + 1] + w01 * row1[i0 + 1] + w11 * row1[i1 + 1];
            float b = w00 * row0[i0 + 2] + w10 * row0[i1 + 2] + w01 * row1[i0 + 2] + w11 * row1[i1 + 2];

            d0[x] = (float)((int)r) * inv255;
            d1[x] = (float)((int)g) * inv255;
            d2[x] = (float)((int)b) * inv255;
        }
    }
}

// Calculate Intersection over Union (IoU)
float calculate_iou(const BoundingBox& b1, const BoundingBox& b2) {
    int x1_inter = std::max(b1.x1, b2.x1);
    int y1_inter = std::max(b1.y1, b2.y1);
    int x2_inter = std::min(b1.x2, b2.x2);
    int y2_inter = std::min(b1.y2, b2.y2);
    
    if (x2_inter <= x1_inter || y2_inter <= y1_inter) return 0.0f;
    
    int inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter);
    int union_area = b1.area() + b2.area() - inter_area;
    return (float)inter_area / union_area;
}

// Draw Bounding Box & Text on Image
void draw_box_text(unsigned char* img, int w, int h, int channels,
                   const BoundingBox& box, const std::string& text) {
    // Clamp box coordinates
    int x1 = std::max(0, std::min(box.x1, w - 1));
    int y1 = std::max(0, std::min(box.y1, h - 1));
    int x2 = std::max(0, std::min(box.x2, w - 1));
    int y2 = std::max(0, std::min(box.y2, h - 1));
    
    // Green color for the border
    unsigned char r = 0, g = 255, b = 0;
    int thickness = 3;
    
    // Draw lines
    for (int t = 0; t < thickness; ++t) {
        // Horizontal lines
        for (int x = std::max(0, x1 - t); x <= std::min(w - 1, x2 + t); ++x) {
            if (y1 - t >= 0) {
                img[((y1 - t) * w + x) * channels + 0] = r;
                img[((y1 - t) * w + x) * channels + 1] = g;
                img[((y1 - t) * w + x) * channels + 2] = b;
            }
            if (y2 + t < h) {
                img[((y2 + t) * w + x) * channels + 0] = r;
                img[((y2 + t) * w + x) * channels + 1] = g;
                img[((y2 + t) * w + x) * channels + 2] = b;
            }
        }
        // Vertical lines
        for (int y = std::max(0, y1 - t); y <= std::min(h - 1, y2 + t); ++y) {
            if (x1 - t >= 0) {
                img[(y * w + (x1 - t)) * channels + 0] = r;
                img[(y * w + (x1 - t)) * channels + 1] = g;
                img[(y * w + (x1 - t)) * channels + 2] = b;
            }
            if (x2 + t < w) {
                img[(y * w + (x2 + t)) * channels + 0] = r;
                img[(y * w + (x2 + t)) * channels + 1] = g;
                img[(y * w + (x2 + t)) * channels + 2] = b;
            }
        }
    }
}

#include "layer.h"

class ArgMax_custom : public ncnn::Layer {
public:
    ArgMax_custom() {
        one_blob_only = true;
        support_inplace = false;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int dims = bottom_blob.dims;

        if (dims == 3) {
            top_blob.create(1, h, channels, sizeof(float), opt.blob_allocator);
            if (top_blob.empty()) return -100;

            for (int q = 0; q < channels; q++) {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                for (int y = 0; y < h; y++) {
                    float max_val = -999999.0f;
                    int max_idx = 0;
                    for (int x = 0; x < w; x++) {
                        float val = ptr[y * w + x];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = x;
                        }
                    }
                    outptr[y] = (float)max_idx;
                }
            }
        } else if (dims == 2) {
            top_blob.create(1, h, sizeof(float), opt.blob_allocator);
            if (top_blob.empty()) return -100;

            for (int y = 0; y < h; y++) {
                const float* ptr = bottom_blob.row(y);
                float* outptr = (float*)top_blob + y;
                float max_val = -999999.0f;
                int max_idx = 0;
                for (int x = 0; x < w; x++) {
                    if (ptr[x] > max_val) {
                        max_val = ptr[x];
                        max_idx = x;
                    }
                }
                *outptr = (float)max_idx;
            }
        } else {
            return -1;
        }

        return 0;
    }
};

ncnn::Layer* ArgMax_custom_creator(void* /*userdata*/) {
    return new ArgMax_custom;
}

class NonMaxSuppression_custom : public ncnn::Layer {
public:
    NonMaxSuppression_custom() {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {
        const ncnn::Mat& boxes = bottom_blobs[0];
        const ncnn::Mat& scores = bottom_blobs[1];
        
        if (g_verbose) {
            std::cout << "[DEBUG NMS] boxes shape: w=" << boxes.w << ", h=" << boxes.h << ", dims=" << boxes.dims << std::endl;
            std::cout << "[DEBUG NMS] scores shape: w=" << scores.w << ", h=" << scores.h << ", dims=" << scores.dims << std::endl;
        }

        int max_output_boxes = 100;
        if (bottom_blobs.size() > 2 && !bottom_blobs[2].empty()) {
            float val = ((const float*)bottom_blobs[2])[0];
            if (val < 1e-5f && val > 0.0f) {
                max_output_boxes = *(const int*)&val;
            } else {
                max_output_boxes = (int)val;
            }
        }
        float iou_threshold = 0.45f;
        if (bottom_blobs.size() > 3 && !bottom_blobs[3].empty()) {
            iou_threshold = ((const float*)bottom_blobs[3])[0];
        }
        float score_threshold = 0.25f;
        if (bottom_blobs.size() > 4 && !bottom_blobs[4].empty()) {
            score_threshold = ((const float*)bottom_blobs[4])[0];
        }

        struct Box {
            int index;
            float x1, y1, x2, y2;
            float score;
        };

        std::vector<Box> candidates;
        int num_boxes = boxes.h;
        
        const float* score_ptr = scores;
        
        if (g_verbose) {
            std::cout << "[DEBUG NMS] boxes.cstep=" << boxes.cstep << ", scores.cstep=" << scores.cstep << std::endl;
        }

        for (int i = 0; i < num_boxes; i++) {
            float score = score_ptr[i];
            if (score < score_threshold) continue;

            const float* box_row = boxes.row(i);
            Box b;
            b.index = i;
            b.x1 = box_row[0];
            b.y1 = box_row[1];
            b.x2 = box_row[2];
            b.y2 = box_row[3];
            b.score = score;
            candidates.push_back(b);
            
            if (g_verbose) {
                std::cout << "  [Candidate candidate_idx=" << candidates.size()-1 << " (orig=" << i << ")] score=" << score 
                          << " box=[" << b.x1 << ", " << b.y1 << ", " << b.x2 << ", " << b.y2 << "]" << std::endl;
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Box& a, const Box& b) {
            return a.score > b.score;
        });

        std::vector<Box> selected;
        for (const auto& cand : candidates) {
            if ((int)selected.size() >= max_output_boxes) break;
            bool keep = true;
            if (g_verbose) {
                std::cout << "  Checking candidate (orig=" << cand.index << ") score=" << cand.score << std::endl;
            }
            for (const auto& sel : selected) {
                float x1_inter = std::max(cand.x1, sel.x1);
                float y1_inter = std::max(cand.y1, sel.y1);
                float x2_inter = std::min(cand.x2, sel.x2);
                float y2_inter = std::min(cand.y2, sel.y2);
                
                float inter_area = 0.0f;
                if (x2_inter > x1_inter && y2_inter > y1_inter) {
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter);
                }
                
                float cand_area = (cand.x2 - cand.x1) * (cand.y2 - cand.y1);
                float sel_area = (sel.x2 - sel.x1) * (sel.y2 - sel.y1);
                float union_area = cand_area + sel_area - inter_area;
                
                float iou = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
                if (iou > iou_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push_back(cand);
            }
        }

        int num_selected = selected.size();
        if (g_verbose) {
            std::cout << "[DEBUG NMS] candidates count: " << candidates.size() << ", selected count: " << num_selected << std::endl;
        }
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(3, num_selected, sizeof(float), opt.blob_allocator);
        if (top_blob.empty()) return -100;

        for (int i = 0; i < num_selected; i++) {
            float* row_ptr = top_blob.row(i);
            row_ptr[0] = 0.0f;
            row_ptr[1] = 0.0f;
            row_ptr[2] = (float)selected[i].index;
        }

        return 0;
    }
};

ncnn::Layer* NonMaxSuppression_custom_creator(void* /*userdata*/) {
    return new NonMaxSuppression_custom;
}

class Gather_custom : public ncnn::Layer {
public:
    Gather_custom() {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {
        const ncnn::Mat& data = bottom_blobs[0];
        const ncnn::Mat& indices = bottom_blobs[1];
        
        int num_indices = indices.w * indices.h * indices.c;
        if (g_verbose) {
            std::cout << "[DEBUG Gather] data shape: w=" << data.w << ", h=" << data.h << ", c=" << data.c << ", dims=" << data.dims 
                      << " | indices shape: w=" << indices.w << ", h=" << indices.h << ", c=" << indices.c << ", dims=" << indices.dims 
                      << " | num_indices: " << num_indices << std::endl;
        }
        const float* indices_ptr = indices;

        ncnn::Mat& top_blob = top_blobs[0];
        if (num_indices == 0) {
            top_blob = ncnn::Mat();
            return 0;
        }
        int w = data.w;
        int h = data.h;
        int channels = data.c;
        
        if (data.dims == 2) {
            top_blob.create(w, num_indices, sizeof(float), opt.blob_allocator);
            if (top_blob.empty()) return -100;
            
            for (int i = 0; i < num_indices; i++) {
                int idx = (int)indices_ptr[i];
                if (idx < 0 || idx >= h) {
                    memset(top_blob.row(i), 0, w * sizeof(float));
                } else {
                    memcpy(top_blob.row(i), data.row(idx), w * sizeof(float));
                }
            }
        } else if (data.dims == 1) {
            top_blob.create(num_indices, sizeof(float), opt.blob_allocator);
            if (top_blob.empty()) return -100;
            
            float* out_ptr = top_blob;
            for (int i = 0; i < num_indices; i++) {
                int idx = (int)indices_ptr[i];
                if (idx < 0 || idx >= data.w) {
                    out_ptr[i] = 0.0f;
                } else {
                    out_ptr[i] = data[idx];
                }
            }
        } else if (data.dims == 3) {
            top_blob.create(w, h, num_indices, sizeof(float), opt.blob_allocator);
            if (top_blob.empty()) return -100;
            
            for (int i = 0; i < num_indices; i++) {
                int idx = (int)indices_ptr[i];
                if (idx < 0 || idx >= channels) {
                    memset(top_blob.channel(i), 0, w * h * sizeof(float));
                } else {
                    memcpy(top_blob.channel(i), data.channel(idx), w * h * sizeof(float));
                }
            }
        } else {
            return -1;
        }

        return 0;
    }
};

ncnn::Layer* Gather_custom_creator(void* /*userdata*/) {
    return new Gather_custom;
}

class Shape_custom : public ncnn::Layer {
public:
    Shape_custom() {
        one_blob_only = true;
        support_inplace = false;
    }
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
        if (g_verbose) {
            std::cout << "[DEBUG Shape] input shape: w=" << bottom_blob.w << " h=" << bottom_blob.h << " c=" << bottom_blob.c << " dims=" << bottom_blob.dims << std::endl;
        }
        top_blob.create(4, sizeof(float), opt.blob_allocator);
        if (top_blob.empty()) {
            if (g_verbose) {
                std::cout << "[DEBUG Shape] top_blob creation failed!" << std::endl;
            }
            return -100;
        }
        
        float* ptr = top_blob;
        ptr[0] = 1.0f;
        ptr[1] = (float)bottom_blob.c;
        ptr[2] = (float)bottom_blob.h;
        ptr[3] = (float)bottom_blob.w;
        if (g_verbose) {
            std::cout << "[DEBUG Shape] output: " << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3] << std::endl;
        }
        return 0;
    }
};

ncnn::Layer* Shape_custom_creator(void* /*userdata*/) {
    return new Shape_custom;
}

struct InferenceOutput {
    std::vector<Detection> detections;
    std::vector<unsigned char> jpeg;
    int width;
    int height;
    double inference_ms;
    double total_ms;
};

static double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                         const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

static std::string join_path(const std::string& base, const std::string& file) {
    if (base.empty() || base == ".") return file;
    char last = base[base.size() - 1];
    return base + ((last == '/' || last == '\\') ? "" : "/") + file;
}

static bool read_file(const std::string& path, std::vector<unsigned char>& data) {
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file) return false;
    file.seekg(0, std::ios::end);
    std::streamoff size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) return false;
    data.resize((size_t)size);
    file.read((char*)data.data(), size);
    return file.good() || file.eof();
}

static void jpeg_write_callback(void* context, void* data, int size) {
    std::vector<unsigned char>* output = (std::vector<unsigned char>*)context;
    unsigned char* bytes = (unsigned char*)data;
    output->insert(output->end(), bytes, bytes + size);
}

static int get_optimal_threads() {
#if defined(__APPLE__)
    int perf_cores = 0;
    size_t len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &len, NULL, 0) == 0 && perf_cores > 0) {
        return perf_cores;
    }
#endif
    int big = ncnn::get_big_cpu_count();
    return (big > 0) ? big : std::max(1, ncnn::get_cpu_count());
}

struct MappedFile {
    const unsigned char* data;
    size_t size;
    MappedFile() : data(NULL), size(0) {}
    ~MappedFile() { release(); }
    bool map(const std::string& path) {
        release();
#if defined(__APPLE__) || defined(__linux__)
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            close(fd);
            return false;
        }
        size = (size_t)st.st_size;
        void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) {
            size = 0;
            return false;
        }
        data = (const unsigned char*)p;
        return true;
#else
        return false;
#endif
    }
    void release() {
#if defined(__APPLE__) || defined(__linux__)
        if (data && size > 0) {
            munmap((void*)data, size);
        }
#endif
        data = NULL;
        size = 0;
    }
};

// Normalize and validate Indonesian license plate format: [A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}
static bool normalize_indonesian_plate(std::string& plate) {
    if (plate.size() < 3 || plate.size() > 9) return false;

    auto digit_to_letter = [](char c) -> char {
        switch (c) {
            case '0': return 'D';
            case '1': return 'I';
            case '2': return 'Z';
            case '4': return 'A';
            case '5': return 'S';
            case '6': return 'G';
            case '8': return 'B';
            default: return c;
        }
    };
    auto letter_to_digit = [](char c) -> char {
        switch (c) {
            case 'O': case 'D': case 'Q': return '0';
            case 'I': case 'L': return '1';
            case 'Z': return '2';
            case 'S': return '5';
            case 'G': return '6';
            case 'B': return '8';
            default: return c;
        }
    };

    // First character of an Indonesian plate is always a letter (Kode Wilayah)
    if (plate[0] >= '0' && plate[0] <= '9') {
        plate[0] = digit_to_letter(plate[0]);
    }
    if (plate[0] < 'A' || plate[0] > 'Z') return false;

    size_t n = plate.size();
    size_t idx = 1;
    if (idx < n && plate[idx] >= 'A' && plate[idx] <= 'Z') {
        ++idx;
    }

    // Next 1..4 characters must be digits (Nomor Polisi)
    size_t digit_start = idx;
    while (idx < n && (idx - digit_start) < 4) {
        if (plate[idx] >= '0' && plate[idx] <= '9') {
            ++idx;
        } else if (idx == digit_start) {
            char d = letter_to_digit(plate[idx]);
            if (d >= '0' && d <= '9') {
                plate[idx] = d;
                ++idx;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    size_t digit_count = idx - digit_start;
    if (digit_count < 1 || digit_count > 4) return false;

    // Remaining 0..3 characters must be letters (Seri Akhir)
    size_t suffix_count = n - idx;
    if (suffix_count > 3) return false;
    for (size_t k = idx; k < n; ++k) {
        if (plate[k] >= '0' && plate[k] <= '9') {
            plate[k] = digit_to_letter(plate[k]);
        }
        if (plate[k] < 'A' || plate[k] > 'Z') return false;
    }
    return true;
}

class AlprEngine {
public:
    AlprEngine() : loaded_(false), model_load_ms_(0.0) {
        blob_pool_.set_size_compare_ratio(0.0f);
        workspace_pool_.set_size_compare_ratio(0.0f);
    }

    bool load(const std::string& model_dir, std::string& error) {
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        const int num_threads = get_optimal_threads();

        if (g_verbose) {
            std::cout << "[+] Loading YOLOv9 model (threads=" << num_threads << ")..." << std::endl;
        }
        yolo_net_.opt.num_threads = num_threads;
        yolo_net_.opt.blob_allocator = &blob_pool_;
        yolo_net_.opt.workspace_allocator = &workspace_pool_;
        yolo_net_.opt.lightmode = true;
        yolo_net_.opt.use_fp16_packed = true;
        yolo_net_.opt.use_fp16_storage = true;
        yolo_net_.opt.use_fp16_arithmetic = true;

        const std::string yolo_param_path = join_path(model_dir, "yolo.param");
        const std::string yolo_bin_path = join_path(model_dir, "yolo.bin");
        if (yolo_net_.load_param(yolo_param_path.c_str()) != 0) {
            error = "Failed to load YOLOv9 param";
            return false;
        }
        if (yolo_bin_map_.map(yolo_bin_path)) {
            if (yolo_net_.load_model(yolo_bin_map_.data) == 0) {
                error = "Failed to load YOLOv9 mapped model";
                return false;
            }
        } else if (yolo_net_.load_model(yolo_bin_path.c_str()) != 0) {
            error = "Failed to load YOLOv9 model";
            return false;
        }

        if (g_verbose) {
            std::cout << "[+] Loading CCT-XS OCR model..." << std::endl;
        }
        ocr_net_.opt.num_threads = num_threads;
        ocr_net_.opt.blob_allocator = &blob_pool_;
        ocr_net_.opt.workspace_allocator = &workspace_pool_;
        ocr_net_.opt.lightmode = true;
        ocr_net_.opt.use_fp16_packed = true;
        ocr_net_.opt.use_fp16_storage = true;
        ocr_net_.opt.use_fp16_arithmetic = true;

        const std::string ocr_param_path = join_path(model_dir, "ocr.param");
        const std::string ocr_bin_path = join_path(model_dir, "ocr_patched.bin");
        if (ocr_net_.load_param(ocr_param_path.c_str()) != 0) {
            error = "Failed to load OCR param";
            return false;
        }
        if (ocr_bin_map_.map(ocr_bin_path)) {
            if (ocr_net_.load_model(ocr_bin_map_.data) == 0) {
                error = "Failed to load OCR mapped model";
                return false;
            }
        } else if (ocr_net_.load_model(ocr_bin_path.c_str()) != 0) {
            error = "Failed to load OCR model";
            return false;
        }

        // Precompute static token reducer query blobs once at startup
        {
            ncnn::Mat dummy(128, 64, 3);
            dummy.fill(0.0f);
            ncnn::Extractor ex = ocr_net_.create_extractor();
            ex.set_light_mode(false);
            ex.input("CCT_OCR_1/conv_stem_1/conv2d_1/BiasAdd__366:0", dummy);
            ex.extract("CCT_OCR_1/token_reducer_1/Tile/ReadVariableOp:0_splitncnn_0", token_tile_const_);
            ex.extract("CCT_OCR_1/token_reducer_1/multi_head_attention_3_1/transpose_1:0", token_query_const_);
            token_tile_const_ = token_tile_const_.clone();
            token_query_const_ = token_query_const_.clone();
        }
        ncnn::set_omp_num_threads(num_threads);

        loaded_ = true;
        model_load_ms_ = elapsed_ms(started, std::chrono::steady_clock::now());
        if (g_verbose) {
            std::cout << "[+] Models loaded once in " << std::fixed << std::setprecision(2)
                      << model_load_ms_ << " ms" << std::endl;
        }
        return true;
    }

    bool infer(const unsigned char* encoded, size_t encoded_size, InferenceOutput& output, std::string& error) {
        std::lock_guard<std::mutex> guard(infer_mutex_);
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        const std::chrono::steady_clock::time_point total_started = std::chrono::steady_clock::now();
        if (!loaded_) {
            error = "Models are not loaded";
            return false;
        }
        if (!encoded || encoded_size == 0) {
            error = "Image is empty";
            return false;
        }

        int img_w = 0, img_h = 0, img_c = 0;
        unsigned char* img_data = stbi_load_from_memory(encoded, (int)encoded_size, &img_w, &img_h, &img_c, 3);
        if (!img_data) {
            error = "Unsupported or invalid image";
            return false;
        }
        img_c = 3;
        const std::chrono::steady_clock::time_point inference_started = std::chrono::steady_clock::now();

        const int target_w = 416;
        const int target_h = 416;
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        ncnn::Mat yolo_in;
        letterbox_to_yolo_mat(img_data, img_w, img_h, yolo_in, target_w, target_h, scale, pad_x, pad_y, &blob_pool_);

        ncnn::Extractor yolo_ex = yolo_net_.create_extractor();
        yolo_ex.input("images", yolo_in);

        static const char* const score_names[3] = {
            "/model/model.22/cv3.0/cv3.0.2/Conv_output_0",
            "/model/model.22/cv3.1/cv3.1.2/Conv_output_0",
            "/model/model.22/cv3.2/cv3.2.2/Conv_output_0"
        };
        static const char* const box_names[3] = {
            "/model/model.22/cv2.0/cv2.0.2/Conv_output_0",
            "/model/model.22/cv2.1/cv2.1.2/Conv_output_0",
            "/model/model.22/cv2.2/cv2.2.2/Conv_output_0"
        };
        static const int strides[3] = {8, 16, 32};
        const float logit_threshold = -0.4054651f; // -log(1 / 0.40 - 1)

        struct RawBox {
            float x1, y1, x2, y2, score;
        };
        std::vector<RawBox> candidates;

        for (int head = 0; head < 3; ++head) {
            ncnn::Mat sm;
            if (yolo_ex.extract(score_names[head], sm) != 0) {
                stbi_image_free(img_data);
                error = "Failed to extract detector score output";
                return false;
            }
            const float* sp = sm.channel(0);
            int gh = sm.h;
            int gw = sm.w;
            int total_cells = gh * gw;

            bool has_candidate = false;
            for (int idx = 0; idx < total_cells; ++idx) {
                if (sp[idx] >= logit_threshold) {
                    has_candidate = true;
                    break;
                }
            }
            if (!has_candidate) continue;

            ncnn::Mat bm;
            if (yolo_ex.extract(box_names[head], bm) != 0) {
                stbi_image_free(img_data);
                error = "Failed to extract detector box output";
                return false;
            }
            const float stride = (float)strides[head];

            for (int y = 0; y < gh; ++y) {
                for (int x = 0; x < gw; ++x) {
                    int sidx = y * gw + x;
                    float logit = sp[sidx];
                    if (logit < logit_threshold) continue;
                    float score = 1.0f / (1.0f + std::exp(-logit));
                    if (score < 0.40f) continue;

                    float dist[4];
                    for (int side = 0; side < 4; ++side) {
                        float vals[16];
                        float max_v = -1e9f;
                        for (int k = 0; k < 16; ++k) {
                            float v = bm.channel(side * 16 + k)[sidx];
                            vals[k] = v;
                            if (v > max_v) max_v = v;
                        }
                        float sum_exp = 0.0f;
                        float weighted = 0.0f;
                        for (int k = 0; k < 16; ++k) {
                            float e = std::exp(vals[k] - max_v);
                            sum_exp += e;
                            weighted += e * k;
                        }
                        dist[side] = weighted / sum_exp;
                    }

                    float cx = (float)x + 0.5f;
                    float cy = (float)y + 0.5f;
                    RawBox rb;
                    rb.x1 = (cx - dist[0]) * stride;
                    rb.y1 = (cy - dist[1]) * stride;
                    rb.x2 = (cx + dist[2]) * stride;
                    rb.y2 = (cy + dist[3]) * stride;
                    rb.score = score;
                    candidates.push_back(rb);
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const RawBox& a, const RawBox& b) {
            return a.score > b.score;
        });

        std::vector<RawBox> selected;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (selected.size() >= 100) break;
            const RawBox& cand = candidates[i];
            bool keep = true;
            for (size_t j = 0; j < selected.size(); ++j) {
                const RawBox& sel = selected[j];
                float ix1 = std::max(cand.x1, sel.x1);
                float iy1 = std::max(cand.y1, sel.y1);
                float ix2 = std::min(cand.x2, sel.x2);
                float iy2 = std::min(cand.y2, sel.y2);
                float inter = (ix2 > ix1 && iy2 > iy1) ? (ix2 - ix1) * (iy2 - iy1) : 0.0f;
                float u = (cand.x2 - cand.x1) * (cand.y2 - cand.y1) + (sel.x2 - sel.x1) * (sel.y2 - sel.y1) - inter;
                if (u > 0.0f && (inter / u) > 0.45f) {
                    keep = false;
                    break;
                }
            }
            if (keep) selected.push_back(cand);
        }

        std::vector<BoundingBox> boxes;
        boxes.reserve(selected.size());
        for (size_t i = 0; i < selected.size(); ++i) {
            const RawBox& source = selected[i];
            int x1 = (int)std::round((source.x1 - pad_x) / scale);
            int y1 = (int)std::round((source.y1 - pad_y) / scale);
            int x2 = (int)std::round((source.x2 - pad_x) / scale);
            int y2 = (int)std::round((source.y2 - pad_y) / scale);
            int horizontal_pad = (int)std::round((x2 - x1) * 0.08f);
            x1 -= horizontal_pad;
            x2 += horizontal_pad;

            BoundingBox box;
            box.x1 = std::max(0, std::min(x1, img_w - 1));
            box.y1 = std::max(0, std::min(y1, img_h - 1));
            box.x2 = std::max(0, std::min(x2, img_w));
            box.y2 = std::max(0, std::min(y2, img_h));
            box.score = source.score;
            // Indonesian license plates are horizontal rectangles (width > height)
            if (box.area() > 0 && box.width() > box.height()) {
                boxes.push_back(box);
            }
        }

        output.detections.clear();
        output.detections.reserve(boxes.size());
        for (size_t d = 0; d < boxes.size(); ++d) {
            const BoundingBox& box = boxes[d];
            ncnn::Mat ocr_in;
            crop_resize_to_ocr_mat(img_data, img_w, img_h, box.x1, box.y1, box.x2, box.y2, ocr_in, &blob_pool_);

            ncnn::Extractor ocr_ex = ocr_net_.create_extractor();
            ocr_ex.input("CCT_OCR_1/conv_stem_1/conv2d_1/BiasAdd__366:0", ocr_in);
            ocr_ex.input("CCT_OCR_1/token_reducer_1/Tile/ReadVariableOp:0_splitncnn_0", token_tile_const_);
            ocr_ex.input("CCT_OCR_1/token_reducer_1/multi_head_attention_3_1/transpose_1:0", token_query_const_);
            ncnn::Mat ocr_out;
            if (ocr_ex.extract("CCT_OCR_1/plate_1/dense_16_1/BiasAdd:0", ocr_out) != 0) {
                stbi_image_free(img_data);
                error = "Failed to extract OCR output";
                return false;
            }

            std::string text;
            float conf_sum = 0.0f;
            int conf_cnt = 0;
            for (int t = 0; t < ocr_out.h; ++t) {
                const float* row = ocr_out.row(t);
                int max_index = 0;
                float max_logit = row[0];
                for (int c = 1; c < ocr_out.w; ++c) {
                    if (row[c] > max_logit) {
                        max_logit = row[c];
                        max_index = c;
                    }
                }
                if (max_index >= 0 && max_index != BLANK_INDEX && max_index < (int)CHARSET.size()) {
                    float sum_exp = 0.0f;
                    for (int c = 0; c < ocr_out.w; ++c) {
                        sum_exp += std::exp(row[c] - max_logit);
                    }
                    text += CHARSET[max_index];
                    conf_sum += 1.0f / sum_exp;
                    ++conf_cnt;
                }
            }

            if (!normalize_indonesian_plate(text)) {
                continue;
            }

            Detection detection;
            detection.bbox = box;
            detection.text = text;
            detection.ocr_conf = conf_cnt > 0 ? (conf_sum / conf_cnt) : 0.0f;
            output.detections.push_back(detection);
        }

        output.inference_ms = elapsed_ms(inference_started, std::chrono::steady_clock::now());
        for (size_t i = 0; i < output.detections.size(); ++i) {
            draw_box_text(img_data, img_w, img_h, img_c, output.detections[i].bbox, output.detections[i].text);
        }
        output.jpeg.clear();
        output.jpeg.reserve(encoded_size + 16384);
        if (!stbi_write_jpg_to_func(jpeg_write_callback, &output.jpeg, img_w, img_h, img_c, img_data, 90)) {
            stbi_image_free(img_data);
            error = "Failed to encode result image";
            return false;
        }
        stbi_image_free(img_data);

        output.width = img_w;
        output.height = img_h;
        output.total_ms = elapsed_ms(total_started, std::chrono::steady_clock::now());
        return true;
    }

    bool loaded() const { return loaded_; }
    double model_load_ms() const { return model_load_ms_; }

private:
    ncnn::UnlockedPoolAllocator blob_pool_;
    ncnn::PoolAllocator workspace_pool_;
    MappedFile yolo_bin_map_;
    MappedFile ocr_bin_map_;
    ncnn::Mat token_tile_const_;
    ncnn::Mat token_query_const_;
    ncnn::Net yolo_net_;
    ncnn::Net ocr_net_;
    bool loaded_;
    double model_load_ms_;
    std::mutex infer_mutex_;
};

static std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (size_t i = 0; i < value.size(); ++i) {
        unsigned char c = (unsigned char)value[i];
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    out << value[i];
                }
        }
    }
    return out.str();
}

static std::string base64_encode(const std::vector<unsigned char>& data) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        unsigned int value = (unsigned int)data[i] << 16;
        if (i + 1 < data.size()) value |= (unsigned int)data[i + 1] << 8;
        if (i + 2 < data.size()) value |= data[i + 2];
        output += table[(value >> 18) & 63];
        output += table[(value >> 12) & 63];
        output += (i + 1 < data.size()) ? table[(value >> 6) & 63] : '=';
        output += (i + 2 < data.size()) ? table[value & 63] : '=';
    }
    return output;
}

static std::string inference_json(const InferenceOutput& output) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    json << "{\"success\":true,\"inference_ms\":" << output.inference_ms
         << ",\"total_ms\":" << output.total_ms
         << ",\"image_width\":" << output.width
         << ",\"image_height\":" << output.height
         << ",\"detections\":[";
    for (size_t i = 0; i < output.detections.size(); ++i) {
        if (i) json << ',';
        const Detection& detection = output.detections[i];
        json << "{\"plate\":\"" << json_escape(detection.text) << "\""
             << ",\"detection_confidence\":" << detection.bbox.score
             << ",\"ocr_confidence\":" << detection.ocr_conf
             << ",\"bbox\":[" << detection.bbox.x1 << ',' << detection.bbox.y1 << ','
             << detection.bbox.x2 << ',' << detection.bbox.y2 << "]}";
    }
    json << "],\"result_image_base64\":\"" << base64_encode(output.jpeg) << "\"}";
    return json.str();
}

static int run_server(AlprEngine& engine, const std::string& host, int port, const std::string& web_file) {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(1); };
    server.set_payload_max_length(15 * 1024 * 1024);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(30, 0);

    server.Get("/", [web_file](const httplib::Request&, httplib::Response& response) {
        std::vector<unsigned char> html;
        if (!read_file(web_file, html)) {
            response.status = 404;
            response.set_content("Demo page not found", "text/plain");
            return;
        }
        response.set_content(std::string((const char*)html.data(), html.size()), "text/html; charset=utf-8");
    });

    server.Get("/api/health", [&engine](const httplib::Request&, httplib::Response& response) {
        std::ostringstream json;
        json << std::fixed << std::setprecision(2)
             << "{\"status\":\"ok\",\"models_loaded\":" << (engine.loaded() ? "true" : "false")
             << ",\"model_load_ms\":" << engine.model_load_ms() << "}";
        response.set_content(json.str(), "application/json");
    });

    server.Post("/api/infer", [&engine](const httplib::Request& request, httplib::Response& response) {
        if (!request.form.has_file("image")) {
            response.status = 400;
            response.set_content("{\"success\":false,\"error\":\"multipart field 'image' is required\"}", "application/json");
            return;
        }
        const httplib::FormData image = request.form.get_file("image");
        InferenceOutput output;
        std::string error;
        if (!engine.infer((const unsigned char*)image.content.data(), image.content.size(), output, error)) {
            response.status = 422;
            response.set_content("{\"success\":false,\"error\":\"" + json_escape(error) + "\"}", "application/json");
            return;
        }
        response.set_content(inference_json(output), "application/json");
    });

    std::cout << "[+] API listening on http://" << host << ':' << port << std::endl;
    std::cout << "[+] POST /api/infer (multipart field: image)" << std::endl;
    return server.listen(host.c_str(), port) ? 0 : 1;
}

static void print_usage(const char* program) {
    std::cerr << "Usage:\n"
              << "  " << program << " <image_path> [--verbose]\n"
              << "  " << program << " serve [--host 0.0.0.0] [--port 8080] [--models models] [--web web/index.html]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const bool serve = std::string(argv[1]) == "serve";
    std::string image_path;
    std::string host = "0.0.0.0";
    std::string model_dir = "models";
    std::string web_file = "web/index.html";
    int port = 8080;

    for (int i = serve ? 2 : 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            g_verbose = true;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--models" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--web" && i + 1 < argc) {
            web_file = argv[++i];
        } else if (!serve && image_path.empty()) {
            image_path = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    AlprEngine engine;
    std::string error;
    if (!engine.load(model_dir, error)) {
        std::cerr << "[-] " << error << std::endl;
        return 1;
    }

    if (serve) {
        if (port < 1 || port > 65535) {
            std::cerr << "Invalid port" << std::endl;
            return 1;
        }
        return run_server(engine, host, port, web_file);
    }

    std::vector<unsigned char> image;
    if (image_path.empty() || !read_file(image_path, image)) {
        std::cerr << "[-] Failed to read image: " << image_path << std::endl;
        return 1;
    }

    InferenceOutput output;
    if (!engine.infer(image.data(), image.size(), output, error)) {
        std::cerr << "[-] " << error << std::endl;
        return 1;
    }

    std::ofstream result("result.jpg", std::ios::binary);
    result.write((const char*)output.jpeg.data(), output.jpeg.size());
    std::cout << "[+] Found " << output.detections.size() << " license plate(s)" << std::endl;
    for (size_t i = 0; i < output.detections.size(); ++i) {
        const Detection& detection = output.detections[i];
        std::cout << i + 1 << ". " << detection.text
                  << " | detection=" << std::fixed << std::setprecision(2) << detection.bbox.score
                  << " | ocr=" << detection.ocr_conf << std::endl;
    }
    std::cout << "[+] Inference: " << output.inference_ms << " ms | Total: " << output.total_ms << " ms" << std::endl;
    std::cout << "[+] Saved result.jpg" << std::endl;
    return 0;
}
