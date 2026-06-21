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

// Include stb image headers
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

// Simple Bilinear Resizing
void resize_bilinear(const unsigned char* src, int src_w, int src_h, int channels,
                        unsigned char* dst, int dst_w, int dst_h) {
    for (int y = 0; y < dst_h; ++y) {
        float src_y = (float)y * (float)src_h / (float)dst_h;
        int y0 = (int)src_y;
        int y1 = std::min(y0 + 1, src_h - 1);
        float dy = src_y - (float)y0;
        
        for (int x = 0; x < dst_w; ++x) {
            float src_x = (float)x * (float)src_w / (float)dst_w;
            int x0 = (int)src_x;
            int x1 = std::min(x0 + 1, src_w - 1);
            float dx = src_x - (float)x0;
            
            for (int c = 0; c < channels; ++c) {
                float p00 = src[(y0 * src_w + x0) * channels + c];
                float p10 = src[(y0 * src_w + x1) * channels + c];
                float p01 = src[(y1 * src_w + x0) * channels + c];
                float p11 = src[(y1 * src_w + x1) * channels + c];
                
                float val = (1.0f - dx) * (1.0f - dy) * p00 +
                            dx * (1.0f - dy) * p10 +
                            (1.0f - dx) * dy * p01 +
                            dx * dy * p11;
                            
                dst[(y * dst_w + x) * channels + c] = (unsigned char)std::max(0.0f, std::min(val, 255.0f));
            }
        }
    }
}

// Letterboxing for YOLOv9 input (416x416)
void letterbox(const unsigned char* src, int src_w, int src_h, int channels,
               unsigned char* dst, int target_w, int target_h,
               float& scale, int& pad_x, int& pad_y) {
    scale = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = (int)std::round(src_w * scale);
    int new_h = (int)std::round(src_h * scale);
    
    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;
    
    // Fill background with padding color (114, 114, 114)
    std::fill(dst, dst + target_w * target_h * channels, 114);
    
    // Resize image
    std::vector<unsigned char> resized(new_w * new_h * channels);
    resize_bilinear(src, src_w, src_h, channels, resized.data(), new_w, new_h);
    
    // Copy resized image to the center of destination
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            for (int c = 0; c < channels; ++c) {
                int dst_idx = ((y + pad_y) * target_w + (x + pad_x)) * channels + c;
                int src_idx = (y * new_w + x) * channels + c;
                dst[dst_idx] = resized[src_idx];
            }
        }
    }
}

// Crop Bounding Box with Clamping
std::vector<unsigned char> crop_bbox(const unsigned char* src, int src_w, int src_h, int channels,
                                     int x1, int y1, int x2, int y2,
                                     int& crop_w, int& crop_h) {
    x1 = std::max(0, std::min(x1, src_w - 1));
    y1 = std::max(0, std::min(y1, src_h - 1));
    x2 = std::max(0, std::min(x2, src_w));
    y2 = std::max(0, std::min(y2, src_h));
    
    crop_w = x2 - x1;
    crop_h = y2 - y1;
    
    if (crop_w <= 0 || crop_h <= 0) {
        crop_w = 1;
        crop_h = 1;
        return std::vector<unsigned char>(channels, 0);
    }
    
    std::vector<unsigned char> crop_data(crop_w * crop_h * channels);
    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            for (int c = 0; c < channels; ++c) {
                crop_data[(y * crop_w + x) * channels + c] = src[((y + y1) * src_w + (x + x1)) * channels + c];
            }
        }
    }
    return crop_data;
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

class AlprEngine {
public:
    AlprEngine() : loaded_(false), model_load_ms_(0.0) {}

    bool load(const std::string& model_dir, std::string& error) {
        const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

        std::cout << "[+] Loading YOLOv9 model..." << std::endl;
        yolo_net_.opt.use_fp16_storage = false;
        yolo_net_.opt.use_fp16_arithmetic = false;
        yolo_net_.opt.use_fp16_packed = false;
        yolo_net_.register_custom_layer("ArgMax", ArgMax_custom_creator);
        yolo_net_.register_custom_layer("NonMaxSuppression", NonMaxSuppression_custom_creator);
        yolo_net_.register_custom_layer("Gather", Gather_custom_creator);
        if (yolo_net_.load_param(join_path(model_dir, "yolo.param").c_str()) != 0 ||
            yolo_net_.load_model(join_path(model_dir, "yolo.bin").c_str()) != 0) {
            error = "Failed to load YOLOv9 model";
            return false;
        }

        std::cout << "[+] Loading CCT-XS OCR model..." << std::endl;
        ocr_net_.opt.use_fp16_packed = false;
        ocr_net_.opt.use_fp16_storage = false;
        ocr_net_.opt.use_fp16_arithmetic = false;
        ocr_net_.register_custom_layer("Shape", Shape_custom_creator);
        ocr_net_.register_custom_layer("Gather", Gather_custom_creator);
        if (ocr_net_.load_param(join_path(model_dir, "ocr.param").c_str()) != 0 ||
            ocr_net_.load_model(join_path(model_dir, "ocr_patched.bin").c_str()) != 0) {
            error = "Failed to load OCR model";
            return false;
        }

        loaded_ = true;
        model_load_ms_ = elapsed_ms(started, std::chrono::steady_clock::now());
        std::cout << "[+] Models loaded once in " << std::fixed << std::setprecision(2)
                  << model_load_ms_ << " ms" << std::endl;
        return true;
    }

    bool infer(const unsigned char* encoded, size_t encoded_size, InferenceOutput& output, std::string& error) {
        std::lock_guard<std::mutex> guard(infer_mutex_);
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
        std::vector<unsigned char> padded_img(target_w * target_h * img_c);
        letterbox(img_data, img_w, img_h, img_c, padded_img.data(), target_w, target_h, scale, pad_x, pad_y);

        ncnn::Mat yolo_in(target_w, target_h, 3);
        for (int c = 0; c < 3; ++c) {
            float* ptr = yolo_in.channel(c);
            for (int y = 0; y < target_h; ++y) {
                for (int x = 0; x < target_w; ++x) {
                    ptr[y * target_w + x] = padded_img[(y * target_w + x) * img_c + c] / 255.0f;
                }
            }
        }

        ncnn::Extractor yolo_ex = yolo_net_.create_extractor();
        yolo_ex.set_light_mode(false);
        yolo_ex.input("images", yolo_in);
        ncnn::Mat nms_out, boxes_out, scores_out;
        if (yolo_ex.extract("/end2end/NonMaxSuppression_output_0", nms_out) != 0 ||
            yolo_ex.extract("/end2end/Add_output_0", boxes_out) != 0 ||
            yolo_ex.extract("/end2end/Transpose_1_output_0", scores_out) != 0) {
            stbi_image_free(img_data);
            error = "Failed to extract detector output";
            return false;
        }

        std::vector<BoundingBox> boxes;
        const float* scores_ptr = scores_out;
        for (int i = 0; i < nms_out.h; ++i) {
            const float* row = nms_out.row(i);
            int idx = (int)row[2];
            if (idx < 0 || idx >= boxes_out.h) continue;
            float score = scores_ptr[idx];
            if (score < 0.40f) continue;

            const float* source = boxes_out.row(idx);
            int x1 = (int)std::round((source[0] - pad_x) / scale);
            int y1 = (int)std::round((source[1] - pad_y) / scale);
            int x2 = (int)std::round((source[2] - pad_x) / scale);
            int y2 = (int)std::round((source[3] - pad_y) / scale);
            int horizontal_pad = (int)std::round((x2 - x1) * 0.08f);
            x1 -= horizontal_pad;
            x2 += horizontal_pad;

            BoundingBox box;
            box.x1 = std::max(0, std::min(x1, img_w - 1));
            box.y1 = std::max(0, std::min(y1, img_h - 1));
            box.x2 = std::max(0, std::min(x2, img_w));
            box.y2 = std::max(0, std::min(y2, img_h));
            box.score = score;
            if (box.area() > 0) boxes.push_back(box);
        }

        output.detections.clear();
        for (size_t d = 0; d < boxes.size(); ++d) {
            const BoundingBox& box = boxes[d];
            int crop_w = 0, crop_h = 0;
            std::vector<unsigned char> crop = crop_bbox(
                img_data, img_w, img_h, img_c, box.x1, box.y1, box.x2, box.y2, crop_w, crop_h);
            const int ocr_w = 128;
            const int ocr_h = 64;
            std::vector<unsigned char> resized(ocr_w * ocr_h * img_c);
            resize_bilinear(crop.data(), crop_w, crop_h, img_c, resized.data(), ocr_w, ocr_h);

            ncnn::Mat ocr_in(3, 128, 64);
            for (int row = 0; row < 64; ++row) {
                float* ptr = ocr_in.channel(row);
                for (int col = 0; col < 128; ++col) {
                    ptr[col * 3 + 0] = (float)resized[(row * 128 + col) * img_c + 0];
                    ptr[col * 3 + 1] = (float)resized[(row * 128 + col) * img_c + 1];
                    ptr[col * 3 + 2] = (float)resized[(row * 128 + col) * img_c + 2];
                }
            }

            ncnn::Extractor ocr_ex = ocr_net_.create_extractor();
            ocr_ex.set_light_mode(false);
            ocr_ex.input("input", ocr_in);
            ncnn::Mat ocr_out;
            if (ocr_ex.extract("plate", ocr_out) != 0) {
                stbi_image_free(img_data);
                error = "Failed to extract OCR output";
                return false;
            }

            std::string text;
            std::vector<float> confidences;
            for (int t = 0; t < ocr_out.h; ++t) {
                const float* row = ocr_out.row(t);
                int max_index = -1;
                float max_probability = -1.0f;
                for (int c = 0; c < ocr_out.w; ++c) {
                    if (row[c] > max_probability) {
                        max_probability = row[c];
                        max_index = c;
                    }
                }
                if (max_index >= 0 && max_index != BLANK_INDEX && max_index < (int)CHARSET.size()) {
                    text += CHARSET[max_index];
                    confidences.push_back(max_probability);
                }
            }

            float confidence = 0.0f;
            for (size_t i = 0; i < confidences.size(); ++i) confidence += confidences[i];
            if (!confidences.empty()) confidence /= confidences.size();

            Detection detection;
            detection.bbox = box;
            detection.text = text;
            detection.ocr_conf = confidence;
            output.detections.push_back(detection);
        }

        output.inference_ms = elapsed_ms(inference_started, std::chrono::steady_clock::now());
        for (size_t i = 0; i < output.detections.size(); ++i) {
            draw_box_text(img_data, img_w, img_h, img_c, output.detections[i].bbox, output.detections[i].text);
        }
        output.jpeg.clear();
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
