// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "yolov8-pose.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifndef RKNPU1
#include <Float16.h>
#endif

#include <iostream>
#include <cmath>
#include <algorithm>

#include <set>
#include <vector>

namespace {
YoloConfig config;
std::vector<char *> labels;
}

YoloConfig& yolo_config() {
    return config;
}

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len) {
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line) {
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[]) {
    if (config.debug_output) {
        printf("load label %s\n", locationFilename);
    }
    return readLines(locationFilename, label, config.class_count);
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    printf("START nms, validCount = %d, threshold = %f\n", validCount, threshold);
    int removed = 0;
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * YoloConfig::box_record_size + 0];
            float ymin0 = outputLocations[n * YoloConfig::box_record_size + 1];
            float xmax0 = outputLocations[n * YoloConfig::box_record_size + 0] + outputLocations[n * YoloConfig::box_record_size + 2];
            float ymax0 = outputLocations[n * YoloConfig::box_record_size + 1] + outputLocations[n * YoloConfig::box_record_size + 3];

            float xmin1 = outputLocations[m * YoloConfig::box_record_size + 0];
            float ymin1 = outputLocations[m * YoloConfig::box_record_size + 1];
            float xmax1 = outputLocations[m * YoloConfig::box_record_size + 0] + outputLocations[m * YoloConfig::box_record_size + 2];
            float ymax1 = outputLocations[m * YoloConfig::box_record_size + 1] + outputLocations[m * YoloConfig::box_record_size + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                ++removed;
                order[j] = -1;
            }
        }
    }
    printf("FINISH nms, removed: %d\n", removed);
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}



void softmax(float *input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

static int process_i8(int8_t *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    const int input_loc_len = 4 * config.dfl_bins;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < config.class_count; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid(deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < 4; ++i) {
                        softmax(&loc[i * config.dfl_bins], config.dfl_bins);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < config.dfl_bins; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * config.dfl_bins + dfl] * dfl;
                        xywh_[2] += loc[2 * config.dfl_bins + dfl] * dfl;
                        xywh_[3] += loc[3 * config.dfl_bins + dfl] * dfl;
                    }
                    xywh_[0]=(w+0.5)-xywh_[0];
                    xywh_[1]=(h+0.5)-xywh_[1];
                    xywh_[2]=(w+0.5)+xywh_[2];
                    xywh_[3]=(h+0.5)+xywh_[3];
                    xywh[0]=((xywh_[0]+xywh_[2])/2)*stride;
                    xywh[1]=((xywh_[1]+xywh_[3])/2)*stride;
                    xywh[2]=(xywh_[2]-xywh_[0])*stride;
                    xywh[3]=(xywh_[3]-xywh_[1])*stride;
                    xywh[0]=xywh[0]-xywh[2]/2;
                    xywh[1]=xywh[1]-xywh[3]/2;
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

#if defined(RV1106_1103)
static int process_i8_nhwc(int8_t *input, const rknn_tensor_attr& attr,
                           int grid_h, int grid_w, int stride,
                           std::vector<float> &boxes,
                           std::vector<float> &boxScores,
                           std::vector<int> &classId, float threshold,
                           int index) {
    const int input_loc_len = 4 * config.dfl_bins;
    const int channel_count = input_loc_len + config.class_count;
    const int row_width = attr.w_stride != 0
        ? static_cast<int>(attr.w_stride)
        : grid_w;
    const size_t required_size = static_cast<size_t>(grid_h) * row_width *
                                 channel_count * sizeof(int8_t);
    const size_t available_size = attr.size_with_stride != 0
        ? attr.size_with_stride
        : attr.size;
    if (input == nullptr || attr.fmt != RKNN_TENSOR_NHWC ||
        attr.n_dims != 4 || static_cast<int>(attr.dims[1]) != grid_h ||
        static_cast<int>(attr.dims[2]) != grid_w ||
        static_cast<int>(attr.dims[3]) != channel_count ||
        row_width < grid_w || required_size > available_size) {
        fprintf(stderr, "Unexpected RV1106 detection output layout\n");
        return -1;
    }

    int validCount = 0;
    const int8_t threshold_i8 = qnt_f32_to_affine(
        unsigmoid(threshold), attr.zp, attr.scale);
    std::vector<float> loc(input_loc_len);

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            int8_t *pixel = input + (h * row_width + w) * channel_count;
            for (int class_id = 0; class_id < config.class_count; ++class_id) {
                const int8_t class_score = pixel[input_loc_len + class_id];
                if (class_score < threshold_i8) {
                    continue;
                }

                const float box_confidence = sigmoid(
                    deqnt_affine_to_f32(class_score, attr.zp, attr.scale));
                for (int i = 0; i < input_loc_len; ++i) {
                    loc[i] = deqnt_affine_to_f32(pixel[i], attr.zp, attr.scale);
                }
                for (int i = 0; i < 4; ++i) {
                    softmax(loc.data() + i * config.dfl_bins, config.dfl_bins);
                }

                float xyxy[4] = {0, 0, 0, 0};
                for (int dfl = 0; dfl < config.dfl_bins; ++dfl) {
                    xyxy[0] += loc[dfl] * dfl;
                    xyxy[1] += loc[config.dfl_bins + dfl] * dfl;
                    xyxy[2] += loc[2 * config.dfl_bins + dfl] * dfl;
                    xyxy[3] += loc[3 * config.dfl_bins + dfl] * dfl;
                }

                const float x1 = (w + 0.5f - xyxy[0]) * stride;
                const float y1 = (h + 0.5f - xyxy[1]) * stride;
                const float x2 = (w + 0.5f + xyxy[2]) * stride;
                const float y2 = (h + 0.5f + xyxy[3]) * stride;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2 - x1);
                boxes.push_back(y2 - y1);
                boxes.push_back(float(index + h * grid_w + w));
                boxScores.push_back(box_confidence);
                classId.push_back(class_id);
                ++validCount;
            }
        }
    }
    return validCount;
}

static bool read_keypoint_i8_nhwc(const int8_t *input,
                                  const rknn_tensor_attr& attr,
                                  int keypoint, int dimension, int anchor,
                                  int8_t *value) {
    if (input == nullptr || value == nullptr || attr.n_dims != 4 ||
        attr.fmt != RKNN_TENSOR_NHWC ||
        static_cast<int>(attr.dims[3]) != config.keypoint_count ||
        keypoint < 0 || keypoint >= config.keypoint_count ||
        dimension < 0 || dimension >= config.keypoint_dimensions ||
        anchor < 0 || anchor >= config.anchor_count) {
        return false;
    }

    const int row_width = attr.w_stride != 0
        ? static_cast<int>(attr.w_stride)
        : static_cast<int>(attr.dims[2]);
    int offset = -1;
    if (static_cast<int>(attr.dims[1]) == config.keypoint_dimensions &&
        static_cast<int>(attr.dims[2]) == config.anchor_count) {
        // NCHW [N, keypoint, dimension, anchor] converted to
        // NHWC [N, dimension, anchor, keypoint].
        offset = (dimension * row_width + anchor) * config.keypoint_count + keypoint;
    } else if (static_cast<int>(attr.dims[1]) == config.anchor_count &&
               static_cast<int>(attr.dims[2]) == config.keypoint_dimensions) {
        offset = (anchor * row_width + dimension) * config.keypoint_count + keypoint;
    }

    const size_t available_size = attr.size_with_stride != 0
        ? attr.size_with_stride
        : attr.size;
    if (offset < 0 || static_cast<size_t>(offset) >= available_size) {
        return false;
    }
    *value = input[offset];
    return true;
}
#endif


static int process_u8(uint8_t *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    const int input_loc_len = 4 * config.dfl_bins;
    int validCount = 0;

    uint8_t thres_i8 = qnt_f32_to_affine_u8(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < config.class_count; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid(deqnt_affine_u8_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_u8_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < 4; ++i) {
                        softmax(&loc[i * config.dfl_bins], config.dfl_bins);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < config.dfl_bins; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * config.dfl_bins + dfl] * dfl;
                        xywh_[2] += loc[2 * config.dfl_bins + dfl] * dfl;
                        xywh_[3] += loc[3 * config.dfl_bins + dfl] * dfl;
                    }
                    xywh_[0]=(w+0.5)-xywh_[0];
                    xywh_[1]=(h+0.5)-xywh_[1];
                    xywh_[2]=(w+0.5)+xywh_[2];
                    xywh_[3]=(h+0.5)+xywh_[3];
                    xywh[0]=((xywh_[0]+xywh_[2])/2)*stride;
                    xywh[1]=((xywh_[1]+xywh_[3])/2)*stride;
                    xywh[2]=(xywh_[2]-xywh_[0])*stride;
                    xywh[3]=(xywh_[3]-xywh_[1])*stride;
                    xywh[0]=xywh[0]-xywh[2]/2;
                    xywh[1]=xywh[1]-xywh[3]/2;
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_fp32(float *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    const int input_loc_len = 4 * config.dfl_bins;
    int validCount = 0;
    float thres_fp = unsigmoid(threshold);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < config.class_count; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_fp) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ]);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    for (int i = 0; i < 4; ++i) {
                        softmax(&loc[i * config.dfl_bins], config.dfl_bins);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < config.dfl_bins; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * config.dfl_bins + dfl] * dfl;
                        xywh_[2] += loc[2 * config.dfl_bins + dfl] * dfl;
                        xywh_[3] += loc[3 * config.dfl_bins + dfl] * dfl;
                    }
                    xywh_[0]=(w+0.5)-xywh_[0];
                    xywh_[1]=(h+0.5)-xywh_[1];
                    xywh_[2]=(w+0.5)+xywh_[2];
                    xywh_[3]=(h+0.5)+xywh_[3];
                    xywh[0]=((xywh_[0]+xywh_[2])/2)*stride;
                    xywh[1]=((xywh_[1]+xywh_[3])/2)*stride;
                    xywh[2]=(xywh_[2]-xywh_[0])*stride;
                    xywh[3]=(xywh_[3]-xywh_[1])*stride;
                    xywh[0]=xywh[0]-xywh[2]/2;
                    xywh[1]=xywh[1]-xywh[3]/2;
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold,
                 object_detect_result_list *od_results) {
    if (app_ctx == nullptr || outputs == nullptr || letter_box == nullptr ||
        od_results == nullptr || app_ctx->output_attrs == nullptr) {
        fprintf(stderr, "Invalid post_process arguments\n");
        return -1;
    }
    if (config.class_count <= 0 ||
        config.keypoint_count <= 0 ||
        config.keypoint_count > YoloConfig::keypoint_capacity ||
        config.keypoint_dimensions != YoloConfig::keypoint_value_count ||
        config.dfl_bins <= 0 ||
        config.detection_head_count <= 0 ||
        config.detection_head_count >= static_cast<int>(app_ctx->io_num.n_output) ||
        config.keypoint_output_index < 0 ||
        config.keypoint_output_index >= static_cast<int>(app_ctx->io_num.n_output) ||
        config.anchor_count <= 0 ||
        config.max_results <= 0 ||
        config.max_results > YoloConfig::result_capacity) {
        fprintf(stderr, "Invalid YoloConfig\n");
        return -1;
    }
#if defined(RV1106_1103)
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)outputs;
#else
    rknn_output *_outputs = (rknn_output *)outputs;
#endif
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    memset(od_results, 0, sizeof(object_detect_result_list));
    int index = 0;
#if defined(RV1106_1103)
    for (int i = 0; i < config.detection_head_count; ++i) {
        const rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        if (_outputs[i] == nullptr || _outputs[i]->virt_addr == nullptr ||
            attr.type != RKNN_TENSOR_INT8 || attr.fmt != RKNN_TENSOR_NHWC ||
            attr.n_dims != 4) {
            fprintf(stderr, "Invalid RV1106 detection output %d\n", i);
            return -1;
        }
        grid_h = attr.dims[1];
        grid_w = attr.dims[2];
        stride = model_in_h / grid_h;
        const int count = process_i8_nhwc(
            (int8_t *)_outputs[i]->virt_addr, attr,
            grid_h, grid_w, stride, filterBoxes, objProbs,
            classId, conf_threshold, index);
        if (count < 0) {
            return -1;
        }
        validCount += count;
        index += grid_h * grid_w;
    }
#elif defined(RKNPU1)
    for (int i = 0; i < config.detection_head_count; i++) {
        grid_h = app_ctx->output_attrs[i].dims[1];
        grid_w = app_ctx->output_attrs[i].dims[0];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_u8((uint8_t *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        index += grid_h * grid_w;
    }
#else
    for (int i = 0; i < config.detection_head_count; i++) {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_i8((int8_t *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale,index);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        index += grid_h * grid_w;
    }
#endif
    if (index != config.anchor_count) {
        fprintf(stderr, "YoloConfig anchor_count does not match detection heads\n");
        return -1;
    }
    // no object detect
    if (validCount <= 0) {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= config.max_results) {
            continue;
        }
        int n = indexArray[i];
        float x1 = filterBoxes[n * YoloConfig::box_record_size + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * YoloConfig::box_record_size + 1] - letter_box->y_pad;
        float w = filterBoxes[n * YoloConfig::box_record_size + 2];
        float h = filterBoxes[n * YoloConfig::box_record_size + 3];
        int keypoints_index = (int)filterBoxes[n * YoloConfig::box_record_size + 4];

        const int keypoint_output = config.keypoint_output_index;
        for (int j = 0; j < config.keypoint_count; ++j) {
#if defined(RV1106_1103)
            const rknn_tensor_attr& keypoint_attr = app_ctx->output_attrs[keypoint_output];
            if (_outputs[keypoint_output] == nullptr ||
                _outputs[keypoint_output]->virt_addr == nullptr ||
                keypoint_attr.type != RKNN_TENSOR_INT8) {
                fprintf(stderr, "Invalid RV1106 keypoint output\n");
                return -1;
            }
            int8_t values[YoloConfig::keypoint_value_count];
            for (int dimension = 0;
                 dimension < YoloConfig::keypoint_value_count;
                 ++dimension) {
                if (!read_keypoint_i8_nhwc(
                        (const int8_t *)_outputs[keypoint_output]->virt_addr,
                        keypoint_attr, j, dimension, keypoints_index,
                        &values[dimension])) {
                    fprintf(stderr, "Unexpected RV1106 keypoint tensor layout\n");
                    return -1;
                }
            }
            od_results->results[last_count].keypoints[j][0] =
                (deqnt_affine_to_f32(values[0], keypoint_attr.zp,
                                     keypoint_attr.scale) - letter_box->x_pad) /
                letter_box->scale;
            od_results->results[last_count].keypoints[j][1] =
                (deqnt_affine_to_f32(values[1], keypoint_attr.zp,
                                     keypoint_attr.scale) - letter_box->y_pad) /
                letter_box->scale;
            od_results->results[last_count].keypoints[j][2] =
                deqnt_affine_to_f32(values[2], keypoint_attr.zp,
                                    keypoint_attr.scale);
#else
            if (app_ctx->is_quant) {
                #ifdef RKNPU1
                        od_results->results[last_count].keypoints[j][0] = (deqnt_affine_u8_to_f32(((uint8_t *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 0 * config.anchor_count + keypoints_index],
                                app_ctx->output_attrs[keypoint_output].zp, app_ctx->output_attrs[keypoint_output].scale)- letter_box->x_pad)/ letter_box->scale;
                        od_results->results[last_count].keypoints[j][1] = (deqnt_affine_u8_to_f32(((uint8_t *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 1 * config.anchor_count + keypoints_index],
                                app_ctx->output_attrs[keypoint_output].zp, app_ctx->output_attrs[keypoint_output].scale)- letter_box->y_pad)/ letter_box->scale;
                        od_results->results[last_count].keypoints[j][2] = deqnt_affine_u8_to_f32(((uint8_t *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 2 * config.anchor_count + keypoints_index],
                                app_ctx->output_attrs[keypoint_output].zp, app_ctx->output_attrs[keypoint_output].scale);       
                #else
                        od_results->results[last_count].keypoints[j][0] = ((float)((rknpu2::float16 *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 0 * config.anchor_count + keypoints_index]
                                                                        - letter_box->x_pad)/ letter_box->scale;
                        od_results->results[last_count].keypoints[j][1] = ((float)((rknpu2::float16 *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 1 * config.anchor_count + keypoints_index]
                                                                            - letter_box->y_pad)/ letter_box->scale;
                        od_results->results[last_count].keypoints[j][2] = (float)((rknpu2::float16 *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 2 * config.anchor_count + keypoints_index];
                #endif
            }
            else
            {
                od_results->results[last_count].keypoints[j][0] = (((float *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 0 * config.anchor_count + keypoints_index]
                                                                - letter_box->x_pad)/ letter_box->scale;
                od_results->results[last_count].keypoints[j][1] = (((float *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 1 * config.anchor_count + keypoints_index]
                                                                    - letter_box->y_pad)/ letter_box->scale;
                od_results->results[last_count].keypoints[j][2] = ((float *)_outputs[keypoint_output].buf)[j * config.keypoint_dimensions * config.anchor_count + 2 * config.anchor_count + keypoints_index];
            }
#endif
        }

        int id = classId[n];
        float obj_conf = objProbs[i];
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x1+w, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y1+h, 0, model_in_h) / letter_box->scale);
        // od_results->results[last_count].box.angle = angle;
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        printf("Object conf: %f\n", od_results->results[last_count].prop);
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int init_post_process() {
    deinit_post_process();
    labels.assign(config.class_count, nullptr);
    if (config.labels_path.empty()) {
        return 0;
    }

    const int ret = loadLabelName(config.labels_path.c_str(), labels.data());
    if (ret < 0) {
        printf("Load %s failed!\n", config.labels_path.c_str());
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id) {

    if (cls_id < 0 || cls_id >= config.class_count ||
        cls_id >= static_cast<int>(labels.size())) {
        return "null";
    }

    if (labels[cls_id]) {
        return labels[cls_id];
    }

    return "null";
}

void deinit_post_process() {
    for (char *&label : labels) {
        if (label != nullptr) {
            free(label);
            label = nullptr;
        }
    }
    labels.clear();
}
