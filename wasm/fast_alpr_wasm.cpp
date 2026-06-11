#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "net.h"
#include "mat.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

// Global networks
static ncnn::Net* g_yolo_net = nullptr;
static ncnn::Net* g_ocr_net = nullptr;

const std::vector<char> CHARSET = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '_'
};
const int BLANK_INDEX = CHARSET.size() - 1; // 36 = '_'

struct BoundingBox {
    int x1, y1, x2, y2;
    float score;
    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
    int area() const { return width() * height(); }
};

struct WasmAlprResult {
    int x;
    int y;
    int w;
    int h;
    float confidence;
    std::string plate;
    float ocrConfidence;
};

// --- Custom NCNN Layers (Imported from main.cpp / fast_alpr_jni.cpp) ---

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

ncnn::Layer* ArgMax_custom_creator(void*) { return new ArgMax_custom; }

class NonMaxSuppression_custom : public ncnn::Layer {
public:
    NonMaxSuppression_custom() {
        one_blob_only = false;
        support_inplace = false;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -1;
        }
        const ncnn::Mat& boxes = bottom_blobs[0];
        const ncnn::Mat& scores = bottom_blobs[1];

        int max_output_boxes = 100;
        if (bottom_blobs.size() > 2 && !bottom_blobs[2].empty()) {
            float val = ((const float*)bottom_blobs[2])[0];
            max_output_boxes = (val < 1e-5f && val > 0.0f) ? *(const int*)&val : (int)val;
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
        }
        std::sort(candidates.begin(), candidates.end(), [](const Box& a, const Box& b) {
            return a.score > b.score;
        });
        std::vector<Box> selected;
        for (const auto& cand : candidates) {
            if ((int)selected.size() >= max_output_boxes) break;
            bool keep = true;
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
            if (keep) selected.push_back(cand);
        }
        int num_selected = selected.size();
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

ncnn::Layer* NonMaxSuppression_custom_creator(void*) { return new NonMaxSuppression_custom; }

class Gather_custom : public ncnn::Layer {
public:
    Gather_custom() {
        one_blob_only = false;
        support_inplace = false;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {
        if (bottom_blobs.size() < 2 || top_blobs.empty()) {
            return -1;
        }
        const ncnn::Mat& data = bottom_blobs[0];
        const ncnn::Mat& indices = bottom_blobs[1];

        int num_indices = indices.w * indices.h * indices.c;
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
                out_ptr[i] = (idx < 0 || idx >= data.w) ? 0.0f : data[idx];
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

ncnn::Layer* Gather_custom_creator(void*) { return new Gather_custom; }

class Shape_custom : public ncnn::Layer {
public:
    Shape_custom() {
        one_blob_only = true;
        support_inplace = false;
    }
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
        top_blob.create(4, sizeof(float), opt.blob_allocator);
        if (top_blob.empty()) return -100;
        float* ptr = top_blob;
        ptr[0] = 1.0f;
        ptr[1] = (float)bottom_blob.c;
        ptr[2] = (float)bottom_blob.h;
        ptr[3] = (float)bottom_blob.w;
        return 0;
    }
};

ncnn::Layer* Shape_custom_creator(void*) { return new Shape_custom; }

// --- Preprocessing Helpers ---

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

void letterbox(const unsigned char* src, int src_w, int src_h, int channels,
               unsigned char* dst, int target_w, int target_h,
               float& scale, int& pad_x, int& pad_y) {
    scale = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = (int)std::round(src_w * scale);
    int new_h = (int)std::round(src_h * scale);
    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;
    std::fill(dst, dst + target_w * target_h * channels, 114);
    std::vector<unsigned char> resized(new_w * new_h * channels);
    resize_bilinear(src, src_w, src_h, channels, resized.data(), new_w, new_h);
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

// --- WASM Bindings Implementation ---

bool init() {
    // Release existing instances
    if (g_yolo_net) { delete g_yolo_net; g_yolo_net = nullptr; }
    if (g_ocr_net) { delete g_ocr_net; g_ocr_net = nullptr; }

    std::cout << "[WASM] Loading YOLOv9 model from Virtual FS..." << std::endl;
    g_yolo_net = new ncnn::Net();
    g_yolo_net->opt.use_vulkan_compute = false;
    g_yolo_net->opt.use_fp16_storage = false;
    g_yolo_net->opt.use_fp16_arithmetic = false;
    g_yolo_net->opt.use_fp16_packed = false;
    g_yolo_net->register_custom_layer("ArgMax", ArgMax_custom_creator);
    g_yolo_net->register_custom_layer("NonMaxSuppression", NonMaxSuppression_custom_creator);
    g_yolo_net->register_custom_layer("Gather", Gather_custom_creator);

    if (g_yolo_net->load_param("yolo.param") != 0 ||
        g_yolo_net->load_model("yolo.bin") != 0) {
        std::cerr << "[WASM] Failed to load YOLOv9 model!" << std::endl;
        delete g_yolo_net; g_yolo_net = nullptr;
        return false;
    }

    std::cout << "[WASM] Loading CCT-XS OCR model from Virtual FS..." << std::endl;
    g_ocr_net = new ncnn::Net();
    g_ocr_net->opt.use_vulkan_compute = false;
    g_ocr_net->opt.use_fp16_packed = false;
    g_ocr_net->opt.use_fp16_storage = false;
    g_ocr_net->opt.use_fp16_arithmetic = false;
    g_ocr_net->register_custom_layer("Shape", Shape_custom_creator);
    g_ocr_net->register_custom_layer("Gather", Gather_custom_creator);

    if (g_ocr_net->load_param("ocr.param") != 0 ||
        g_ocr_net->load_model("ocr_patched.bin") != 0) {
        std::cerr << "[WASM] Failed to load OCR model!" << std::endl;
        delete g_yolo_net; g_yolo_net = nullptr;
        delete g_ocr_net; g_ocr_net = nullptr;
        return false;
    }

    std::cout << "[WASM] Models loaded successfully" << std::endl;
    return true;
}

std::vector<WasmAlprResult> detect(uintptr_t rgba_ptr, int img_w, int img_h, bool doOcr) {
    std::vector<WasmAlprResult> results;
    if (!g_yolo_net || (doOcr && !g_ocr_net)) {
        std::cerr << "[WASM] Models not initialized!" << std::endl;
        return results;
    }

    const unsigned char* p_rgba = reinterpret_cast<const unsigned char*>(rgba_ptr);
    int img_c = 3;

    // Convert RGBA to RGB raw buffer
    std::vector<unsigned char> img_rgb(img_w * img_h * 3);
    unsigned char* p_rgb = img_rgb.data();
    for (int i = 0; i < img_w * img_h; ++i) {
        p_rgb[i * 3 + 0] = p_rgba[i * 4 + 0]; // R
        p_rgb[i * 3 + 1] = p_rgba[i * 4 + 1]; // G
        p_rgb[i * 3 + 2] = p_rgba[i * 4 + 2]; // B
    }

    // 2. Preprocess for YOLOv9 (416x416 letterbox)
    int target_w = 416;
    int target_h = 416;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    std::vector<unsigned char> padded_img(target_w * target_h * 3);
    letterbox(p_rgb, img_w, img_h, img_c, padded_img.data(), target_w, target_h, scale, pad_x, pad_y);

    ncnn::Mat yolo_in(target_w, target_h, 3);
    for (int c = 0; c < 3; ++c) {
        float* ptr = yolo_in.channel(c);
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                ptr[y * target_w + x] = padded_img[(y * target_w + x) * 3 + c] / 255.0f;
            }
        }
    }

    // 3. Run YOLOv9 Inference
    ncnn::Extractor yolo_ex = g_yolo_net->create_extractor();
    yolo_ex.set_light_mode(false); // MUST be false since we extract multiple outputs
    yolo_ex.input("images", yolo_in);

    ncnn::Mat nms_out;
    ncnn::Mat boxes_out;
    ncnn::Mat scores_out;
    if (yolo_ex.extract("/end2end/NonMaxSuppression_output_0", nms_out) != 0 ||
        yolo_ex.extract("/end2end/Add_output_0", boxes_out) != 0 ||
        yolo_ex.extract("/end2end/Transpose_1_output_0", scores_out) != 0) {
        std::cerr << "[WASM] Failed to extract YOLOv9 outputs!" << std::endl;
        return results;
    }

    std::vector<BoundingBox> kept_bboxes;
    float confidence_threshold = 0.40f;
    const float* scores_ptr = scores_out;

    for (int i = 0; i < nms_out.h; ++i) {
        const float* row = nms_out.row(i);
        int idx = (int)row[2];
        if (idx < 0 || idx >= boxes_out.h) continue;
        float score = scores_ptr[idx];
        if (score < confidence_threshold) continue;

        const float* box_row = boxes_out.row(idx);
        float pad_x1 = box_row[0];
        float pad_y1 = box_row[1];
        float pad_x2 = box_row[2];
        float pad_y2 = box_row[3];

        int x1 = (int)std::round((pad_x1 - pad_x) / scale);
        int y1 = (int)std::round((pad_y1 - pad_y) / scale);
        int x2 = (int)std::round((pad_x2 - pad_x) / scale);
        int y2 = (int)std::round((pad_y2 - pad_y) / scale);

        BoundingBox bbox;
        bbox.x1 = std::max(0, std::min(x1, img_w - 1));
        bbox.y1 = std::max(0, std::min(y1, img_h - 1));
        bbox.x2 = std::max(0, std::min(x2, img_w));
        bbox.y2 = std::max(0, std::min(y2, img_h));
        bbox.score = score;

        if (bbox.area() > 0) {
            kept_bboxes.push_back(bbox);
        }
    }

    // 4. Perform OCR on each bbox sequentially (if requested)
    for (size_t d = 0; d < kept_bboxes.size(); ++d) {
        const auto& box = kept_bboxes[d];
        WasmAlprResult res;
        res.x = box.x1;
        res.y = box.y1;
        res.w = box.width();
        res.h = box.height();
        res.confidence = box.score;
        res.plate = "";
        res.ocrConfidence = 0.0f;

        if (doOcr) {
            int crop_w, crop_h;
            std::vector<unsigned char> crop_data = crop_bbox(p_rgb, img_w, img_h, img_c, box.x1, box.y1, box.x2, box.y2, crop_w, crop_h);

            int ocr_w = 128;
            int ocr_h = 64;
            std::vector<unsigned char> resized_crop(ocr_w * ocr_h * 3);
            resize_bilinear(crop_data.data(), crop_w, crop_h, 3, resized_crop.data(), ocr_w, ocr_h);

            ncnn::Mat ocr_in(3, 128, 64);
            for (int c_idx = 0; c_idx < 64; ++c_idx) {
                float* ptr = ocr_in.channel(c_idx);
                for (int y_idx = 0; y_idx < 128; ++y_idx) {
                    ptr[y_idx * 3 + 0] = (float)resized_crop[(c_idx * 128 + y_idx) * 3 + 0]; // R
                    ptr[y_idx * 3 + 1] = (float)resized_crop[(c_idx * 128 + y_idx) * 3 + 1]; // G
                    ptr[y_idx * 3 + 2] = (float)resized_crop[(c_idx * 128 + y_idx) * 3 + 2]; // B
                }
            }

            ncnn::Extractor ocr_ex = g_ocr_net->create_extractor();
            ocr_ex.set_light_mode(true);
            ocr_ex.input("input", ocr_in);

            ncnn::Mat ocr_out;
            if (ocr_ex.extract("plate", ocr_out) == 0) {
                std::string ocr_text = "";
                std::vector<float> ocr_scores;
                int prev_index = BLANK_INDEX;
                for (int t = 0; t < ocr_out.h; ++t) {
                    const float* row = ocr_out.row(t);
                    int max_index = -1;
                    float max_prob = -1.0f;
                    for (int c = 0; c < ocr_out.w; ++c) {
                        if (row[c] > max_prob) { max_prob = row[c]; max_index = c; }
                    }
                    if (max_index != BLANK_INDEX && max_index != prev_index) {
                        if (max_index < (int)CHARSET.size()) {
                            ocr_text += CHARSET[max_index];
                            ocr_scores.push_back(max_prob);
                        }
                    }
                    prev_index = max_index;
                }
                res.plate = ocr_text;
                float sum = 0.0f;
                for (float s : ocr_scores) sum += s;
                res.ocrConfidence = ocr_scores.empty() ? 0.0f : (sum / ocr_scores.size());
            }
        }
        results.push_back(res);
    }

    return results;
}

// --- Emscripten JS Bindings ---

EMSCRIPTEN_BINDINGS(fast_alpr_wasm) {
    emscripten::value_object<WasmAlprResult>("WasmAlprResult")
        .field("x", &WasmAlprResult::x)
        .field("y", &WasmAlprResult::y)
        .field("w", &WasmAlprResult::w)
        .field("h", &WasmAlprResult::h)
        .field("confidence", &WasmAlprResult::confidence)
        .field("plate", &WasmAlprResult::plate)
        .field("ocrConfidence", &WasmAlprResult::ocrConfidence);

    emscripten::register_vector<WasmAlprResult>("vector<WasmAlprResult>");

    emscripten::function("init", &init);
    emscripten::function("detect", &detect);
}
