#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [--verbose]" << std::endl;
        return -1;
    }
    std::string img_path = "";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            g_verbose = true;
        } else if (img_path.empty()) {
            img_path = arg;
        }
    }
    if (img_path.empty()) {
        std::cerr << "Error: No input image path specified." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image_path> [--verbose]" << std::endl;
        return -1;
    }
    
    // 1. Load models
    std::cout << "[+] Loading YOLOv9 model..." << std::endl;
    ncnn::Net yolo_net;
    yolo_net.opt.use_fp16_storage = false;
    yolo_net.opt.use_fp16_arithmetic = false;
    yolo_net.opt.use_fp16_packed = false;
    yolo_net.register_custom_layer("ArgMax", ArgMax_custom_creator);
    yolo_net.register_custom_layer("NonMaxSuppression", NonMaxSuppression_custom_creator);
    yolo_net.register_custom_layer("Gather", Gather_custom_creator);
    if (yolo_net.load_param("models/yolo.param") != 0 ||
        yolo_net.load_model("models/yolo.bin") != 0) {
        std::cerr << "[-] Failed to load YOLOv9 model!" << std::endl;
        return -1;
    }
    
    if (g_verbose) {
        std::cout << "[DEBUG] YOLOv9 Blobs list:" << std::endl;
        const std::vector<ncnn::Blob>& blobs = yolo_net.blobs();
        for (size_t i = 0; i < blobs.size(); i++) {
            std::cout << "  Blob #" << i << ": name=" << blobs[i].name << " producer=" << blobs[i].producer << std::endl;
        }
        
        std::cout << "[DEBUG] YOLOv9 Layers list:" << std::endl;
        const std::vector<ncnn::Layer*>& layers = yolo_net.layers();
        for (size_t i = 0; i < layers.size(); i++) {
            if (layers[i] == nullptr) {
                std::cout << "  Layer #" << i << ": NULL!" << std::endl;
            } else {
                std::cout << "  Layer #" << i << ": name=" << layers[i]->name << " type=" << layers[i]->type << std::endl;
            }
        }
    }

    std::cout << "[+] Loading CCT-XS OCR model..." << std::endl;
    ncnn::Net ocr_net;
    ocr_net.opt.use_fp16_packed = false;
    ocr_net.opt.use_fp16_storage = false;
    ocr_net.opt.use_fp16_arithmetic = false;
    ocr_net.register_custom_layer("Shape", Shape_custom_creator);
    ocr_net.register_custom_layer("Gather", Gather_custom_creator);
    if (ocr_net.load_param("models/ocr.param") != 0 ||
        ocr_net.load_model("models/ocr_patched.bin") != 0) {
        std::cerr << "[-] Failed to load OCR model!" << std::endl;
        return -1;
    }
    
    // Print constant values using Extractor
    if (g_verbose) {
        ncnn::Extractor ex = ocr_net.create_extractor();
        ex.set_light_mode(false);
        std::vector<std::string> const_names = {
            "const_fold_opt__570",
            "const_fold_opt__573",
            "const_fold_opt__576",
            "const_fold_opt__577",
            "Const__523",
            "const_ends__415"
        };
        for (const auto& name : const_names) {
            ncnn::Mat out;
            int err = ex.extract(name.c_str(), out);
            std::cout << "[DEBUG CONST EXTRACT] " << name << " -> ret=" << err;
            if (err == 0) {
                std::cout << " shape=(" << out.w << "," << out.h << "," << out.c << ") values=[";
                for (int k = 0; k < out.total(); ++k) {
                    std::cout << " " << out[k];
                }
                std::cout << " ]";
            }
            std::cout << std::endl;
        }
    }
    
    // 2. Load image
    std::cout << "[+] Loading image: " << img_path << std::endl;
    int img_w, img_h, img_c;
    unsigned char* img_data = stbi_load(img_path.c_str(), &img_w, &img_h, &img_c, 3); // force 3 channels (RGB)
    if (!img_data) {
        std::cerr << "[-] Failed to load image: " << img_path << std::endl;
        return -1;
    }
    img_c = 3;
    std::cout << "[+] Image dimensions: " << img_w << "x" << img_h << std::endl;
    
    // 3. Preprocess for YOLOv9 (416x416 letterbox, float32, normalized to [0,1])
    std::cout << "[+] Preprocessing for YOLOv9..." << std::endl;
    int target_w = 416;
    int target_h = 416;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    
    std::vector<unsigned char> padded_img(target_w * target_h * img_c);
    letterbox(img_data, img_w, img_h, img_c, padded_img.data(), target_w, target_h, scale, pad_x, pad_y);
    
    // Create NCNN Mat (NCHW format, normalized [0, 1])
    ncnn::Mat yolo_in(target_w, target_h, 3);
    for (int c = 0; c < 3; ++c) {
        float* ptr = yolo_in.channel(c);
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                ptr[y * target_w + x] = padded_img[(y * target_w + x) * img_c + c] / 255.0f;
            }
        }
    }
    
    // 4. Run YOLOv9 inference
    ncnn::Extractor yolo_ex = yolo_net.create_extractor();
    yolo_ex.set_light_mode(false);
    yolo_ex.input("images", yolo_in);
    
    ncnn::Mat nms_out;
    ncnn::Mat boxes_out;
    ncnn::Mat scores_out;
    if (yolo_ex.extract("/end2end/NonMaxSuppression_output_0", nms_out) != 0) {
        std::cerr << "[-] Failed to extract NMS output!" << std::endl;
        return -1;
    }
    if (yolo_ex.extract("/end2end/Add_output_0", boxes_out) != 0) {
        std::cerr << "[-] Failed to extract candidate boxes!" << std::endl;
        return -1;
    }
    if (yolo_ex.extract("/end2end/Transpose_1_output_0", scores_out) != 0) {
        std::cerr << "[-] Failed to extract scores!" << std::endl;
        return -1;
    }
    
    if (g_verbose) {
        std::cout << "[DEBUG] NMS output shape: w=" << nms_out.w << " h=" << nms_out.h << " c=" << nms_out.c << " dims=" << nms_out.dims << std::endl;
    }
    
    // 5. Parse YOLOv9 detections using NMS selected indices
    std::vector<BoundingBox> kept_detections;
    float confidence_threshold = 0.40f;
    const float* scores_ptr = scores_out;
    if (g_verbose) {
        std::cout << "[DEBUG main] scores_out dims=" << scores_out.dims << " w=" << scores_out.w << " h=" << scores_out.h << " c=" << scores_out.c << std::endl;
    }
    
    for (int i = 0; i < nms_out.h; ++i) {
        const float* row = nms_out.row(i);
        // row[0] = batch_index (0)
        // row[1] = class_index (0)
        // row[2] = index of candidate
        int idx = (int)row[2];
        if (idx < 0 || idx >= boxes_out.h) {
            std::cerr << "[WARNING] NMS candidate index " << idx << " out of bounds [0, " << boxes_out.h << ")" << std::endl;
            continue;
        }
        
        float score = scores_ptr[idx];
        if (g_verbose) {
            std::cout << "[DEBUG main] candidate " << i << " (idx=" << idx << ") score=" << score << std::endl;
        }
        if (score < confidence_threshold) continue;
        
        const float* box_row = boxes_out.row(idx);
        float pad_x1 = box_row[0];
        float pad_y1 = box_row[1];
        float pad_x2 = box_row[2];
        float pad_y2 = box_row[3];
        
        // Remove padding and scale back to original image coordinates
        int x1 = (int)std::round((pad_x1 - pad_x) / scale);
        int y1 = (int)std::round((pad_y1 - pad_y) / scale);
        int x2 = (int)std::round((pad_x2 - pad_x) / scale);
        int y2 = (int)std::round((pad_y2 - pad_y) / scale);
        
        // Widen left and right by 8% of the bounding box width
        int bbox_w = x2 - x1;
        int pad_val = (int)std::round(bbox_w * 0.08f);
        x1 -= pad_val;
        x2 += pad_val;
        
        BoundingBox bbox;
        bbox.x1 = std::max(0, std::min(x1, img_w - 1));
        bbox.y1 = std::max(0, std::min(y1, img_h - 1));
        bbox.x2 = std::max(0, std::min(x2, img_w));
        bbox.y2 = std::max(0, std::min(y2, img_h));
        bbox.score = score;
        
        if (g_verbose) {
            std::cout << "  [BBOX] pad_box=[" << pad_x1 << ", " << pad_y1 << ", " << pad_x2 << ", " << pad_y2 << "]"
                      << " scaled_box=[" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]"
                      << " clamped_box=[" << bbox.x1 << ", " << bbox.y1 << ", " << bbox.x2 << ", " << bbox.y2 << "]"
                      << " area=" << bbox.area() << std::endl;
        }
                  
        if (bbox.area() > 0) {
            kept_detections.push_back(bbox);
        }
    }
    
    std::cout << "[+] Found " << kept_detections.size() << " license plate(s)." << std::endl;
    
    // 6. Run OCR on each cropped license plate
    std::vector<Detection> final_results;
    for (size_t d = 0; d < kept_detections.size(); ++d) {
        const auto& box = kept_detections[d];
        std::cout << "\n--- Processing Plate #" << d + 1 << " ---" << std::endl;
        std::cout << "Location: [" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << "], Confidence: " << box.score * 100.0f << "%" << std::endl;
        
        // Crop license plate
        int crop_w, crop_h;
        std::vector<unsigned char> crop_data = crop_bbox(img_data, img_w, img_h, img_c, box.x1, box.y1, box.x2, box.y2, crop_w, crop_h);
        
        // Resize cropped plate to 128x64 (CCT-XS-v2 input size)
        int ocr_w = 128;
        int ocr_h = 64;
        std::vector<unsigned char> resized_crop(ocr_w * ocr_h * img_c);
        resize_bilinear(crop_data.data(), crop_w, crop_h, img_c, resized_crop.data(), ocr_w, ocr_h);
        
        if (g_verbose) {
            std::cout << "[DEBUG] Saving cropped plate to crop.png..." << std::endl;
            stbi_write_png("crop.png", ocr_w, ocr_h, img_c, resized_crop.data(), ocr_w * img_c);
        }
        
        // Create NCNN Mat (w=3, h=128, c=64) matching the NHWC format expected by the model
        // Send RGB directly (matching Rust — NO BGR swap)
        ncnn::Mat ocr_in(3, 128, 64);
        for (int c_idx = 0; c_idx < 64; ++c_idx) {
            float* ptr = ocr_in.channel(c_idx);
            for (int y_idx = 0; y_idx < 128; ++y_idx) {
                ptr[y_idx * 3 + 0] = (float)resized_crop[(c_idx * 128 + y_idx) * img_c + 0]; // R
                ptr[y_idx * 3 + 1] = (float)resized_crop[(c_idx * 128 + y_idx) * img_c + 1]; // G
                ptr[y_idx * 3 + 2] = (float)resized_crop[(c_idx * 128 + y_idx) * img_c + 2]; // B
            }
        }
        
        if (g_verbose) {
            std::cout << "[DEBUG C++] First 5 BGR values of row 0:" << std::endl;
            const float* ptr_val = ocr_in.channel(0);
            for (int i = 0; i < 5; ++i) {
                std::cout << " [" << ptr_val[i * 3 + 0] << " " << ptr_val[i * 3 + 1] << " " << ptr_val[i * 3 + 2] << "]" << std::endl;
            }
        }
        
        // Run OCR inference
        ncnn::Extractor ocr_ex = ocr_net.create_extractor();
        ocr_ex.set_light_mode(false);
        ocr_ex.input("input", ocr_in);
        
        ncnn::Mat ocr_out;
        int ret = ocr_ex.extract("plate", ocr_out);
        
        // Greedy Decoding (no CTC collapsing for multi-head classification)
        // ocr_out shape is (w=37, h=10, c=1), where w is num_classes (37) and h is seq_len (10)
        std::string ocr_text = "";
        std::vector<float> ocr_scores;
        
        for (int t = 0; t < ocr_out.h; ++t) {
            const float* row = ocr_out.row(t);
            int max_index = -1;
            float max_prob = -1.0f;
            
            for (int c = 0; c < ocr_out.w; ++c) {
                if (row[c] > max_prob) {
                    max_prob = row[c];
                    max_index = c;
                }
            }
            
            if (g_verbose) {
                std::cout << "  t=" << t << ": argmax=" << max_index << " value=" << max_prob 
                          << " (char=" << (max_index < (int)CHARSET.size() ? std::string(1, CHARSET[max_index]) : "<BLANK>") << ")" << std::endl;
             }
            
            // Only skip blank characters
            if (max_index != BLANK_INDEX) {
                if (max_index < (int)CHARSET.size()) {
                    ocr_text += CHARSET[max_index];
                    ocr_scores.push_back(max_prob);
                }
            }
        }
        
        float avg_ocr_conf = 0.0f;
        if (!ocr_scores.empty()) {
            float sum = 0.0f;
            for (float s : ocr_scores) sum += s;
            avg_ocr_conf = sum / ocr_scores.size();
        }
        
        std::cout << "OCR Result: \"" << ocr_text << "\" (Avg Confidence: " << avg_ocr_conf * 100.0f << "%)" << std::endl;
        
        Detection result;
        result.bbox = box;
        result.text = ocr_text;
        result.ocr_conf = avg_ocr_conf;
        final_results.push_back(result);
    }
    
    // 7. Render outputs on the original image and save
    if (!final_results.empty()) {
        std::cout << "\n[+] Drawing results on image..." << std::endl;
        for (const auto& res : final_results) {
            draw_box_text(img_data, img_w, img_h, img_c, res.bbox, res.text);
        }
        
        std::string out_path = "result.jpg";
        std::cout << "[+] Saving result to: " << out_path << std::endl;
        if (stbi_write_jpg(out_path.c_str(), img_w, img_h, img_c, img_data, 90) == 0) {
            std::cerr << "[-] Failed to save result image!" << std::endl;
        }
    }
    
    // Clean up memory
    stbi_image_free(img_data);
    
    std::cout << "\n=== Summary Detections ===" << std::endl;
    for (size_t i = 0; i < final_results.size(); ++i) {
        const auto& res = final_results[i];
        std::cout << i + 1 << ". Bbox: [" << res.bbox.x1 << ", " << res.bbox.y1 << ", " << res.bbox.x2 << ", " << res.bbox.y2 
                  << "] | Score: " << std::fixed << std::setprecision(2) << res.bbox.score 
                  << " | Plate: \"" << res.text << "\" (OCR Conf: " << res.ocr_conf << ")" << std::endl;
    }
    
    return 0;
}
