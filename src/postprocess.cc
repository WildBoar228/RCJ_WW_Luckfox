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

#include "rcj_neurovision.hpp"
#include "postprocess.h"

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
#include <cstring>

#include <set>
#include <vector>

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
            float xmin0 = outputLocations[n * 5 + 0];
            float ymin0 = outputLocations[n * 5 + 1];
            float xmax0 = outputLocations[n * 5 + 0] + outputLocations[n * 5 + 2];
            float ymax0 = outputLocations[n * 5 + 1] + outputLocations[n * 5 + 3];

            float xmin1 = outputLocations[m * 5 + 0];
            float ymin1 = outputLocations[m * 5 + 1];
            float xmax1 = outputLocations[m * 5 + 0] + outputLocations[m * 5 + 2];
            float ymax1 = outputLocations[m * 5 + 1] + outputLocations[m * 5 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
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
    int input_loc_len = 64;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < ww::vision::kClassNum; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid(deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
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


static int process_u8(uint8_t *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    int input_loc_len = 64;
    int validCount = 0;

    uint8_t thres_i8 = qnt_f32_to_affine_u8(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < ww::vision::kClassNum; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid(deqnt_affine_u8_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_u8_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
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
    int input_loc_len = 64;
    int validCount = 0;
    float thres_fp = unsigmoid(threshold);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < ww::vision::kClassNum; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_fp) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ]);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
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

static float output_value(
    const rknn_output& output,
    const rknn_tensor_attr& attr,
    int index
) {
    if (output.want_float) {
        return static_cast<const float*>(output.buf)[index];
    }

    switch (attr.type) {
    case RKNN_TENSOR_INT8:
        return deqnt_affine_to_f32(
            static_cast<const int8_t*>(output.buf)[index],
            attr.zp,
            attr.scale
        );
    case RKNN_TENSOR_UINT8:
        return deqnt_affine_u8_to_f32(
            static_cast<const uint8_t*>(output.buf)[index],
            attr.zp,
            attr.scale
        );
#ifndef RKNPU1
    case RKNN_TENSOR_FLOAT16:
        return static_cast<float>(
            static_cast<const rknpu2::float16*>(output.buf)[index]);
#endif
    case RKNN_TENSOR_FLOAT32:
        return static_cast<const float*>(output.buf)[index];
    default:
        return 0.0f;
    }
}

int post_process(
    ww::vision::ModelHandler* app_ctx,
    void* outputs,
    ww::vision::LetterboxResult* letter_box,
    float conf_threshold,
    float nms_threshold,
    object_detect_result_list* od_results
) {
    if (app_ctx == nullptr || outputs == nullptr ||
        letter_box == nullptr || od_results == nullptr) {
        return -1;
    }

    auto* rknn_outputs = static_cast<rknn_output*>(outputs);
    std::memset(od_results, 0, sizeof(*od_results));

    std::vector<float> filter_boxes;
    std::vector<float> object_probabilities;
    std::vector<int> class_ids;
    int valid_count = 0;
    int anchor_offset = 0;

    for (int i = 0; i < ww::vision::kDetectionHeadNum; ++i) {
        const auto& attr = app_ctx->output_attrs[i];
        const int grid_h = attr.dims[2];
        const int grid_w = attr.dims[3];
        const int stride = app_ctx->input_height / grid_h;

        if (!app_ctx->is_quant) {
            valid_count += process_fp32(
                static_cast<float*>(rknn_outputs[i].buf),
                grid_h, grid_w, stride,
                filter_boxes, object_probabilities, class_ids,
                conf_threshold, attr.zp, attr.scale, anchor_offset
            );
        } else if (attr.type == RKNN_TENSOR_INT8) {
            valid_count += process_i8(
                static_cast<int8_t*>(rknn_outputs[i].buf),
                grid_h, grid_w, stride,
                filter_boxes, object_probabilities, class_ids,
                conf_threshold, attr.zp, attr.scale, anchor_offset
            );
        } else if (attr.type == RKNN_TENSOR_UINT8) {
            valid_count += process_u8(
                static_cast<uint8_t*>(rknn_outputs[i].buf),
                grid_h, grid_w, stride,
                filter_boxes, object_probabilities, class_ids,
                conf_threshold, attr.zp, attr.scale, anchor_offset
            );
        } else {
            return -1;
        }

        anchor_offset += grid_h * grid_w;
    }

    if (valid_count <= 0) {
        return 0;
    }

    std::vector<int> ordered_indices;
    ordered_indices.reserve(valid_count);
    for (int i = 0; i < valid_count; ++i) {
        ordered_indices.push_back(i);
    }
    quick_sort_indice_inverse(
        object_probabilities, 0, valid_count - 1, ordered_indices);

    const std::set<int> classes(class_ids.begin(), class_ids.end());
    for (const int class_id : classes) {
        nms(
            valid_count,
            filter_boxes,
            class_ids,
            ordered_indices,
            class_id,
            nms_threshold
        );
    }

    const int keypoint_output = ww::vision::kDetectionHeadNum;
    const auto& keypoint_attr = app_ctx->output_attrs[keypoint_output];
    const int anchor_count = keypoint_attr.dims[3];

    for (int i = 0;
         i < valid_count && od_results->count < OBJ_NUMB_MAX_SIZE;
         ++i) {
        if (ordered_indices[i] == -1) {
            continue;
        }

        const int candidate_index = ordered_indices[i];
        const int keypoint_index = static_cast<int>(
            filter_boxes[candidate_index * 5 + 4]);
        if (keypoint_index < 0 || keypoint_index >= anchor_count) {
            continue;
        }

        auto& result = od_results->results[od_results->count];
        for (int keypoint = 0;
             keypoint < ww::vision::kKeypointNum;
             ++keypoint) {
            const int base =
                keypoint * ww::vision::kKeypointDimensions * anchor_count;
            const float x = output_value(
                rknn_outputs[keypoint_output], keypoint_attr,
                base + keypoint_index);
            const float y = output_value(
                rknn_outputs[keypoint_output], keypoint_attr,
                base + anchor_count + keypoint_index);
            const float confidence = output_value(
                rknn_outputs[keypoint_output], keypoint_attr,
                base + 2 * anchor_count + keypoint_index);

            result.keypoints[keypoint][0] =
                (x - letter_box->pad_x) / letter_box->scale;
            result.keypoints[keypoint][1] =
                (y - letter_box->pad_y) / letter_box->scale;
            result.keypoints[keypoint][2] = confidence;
        }

        result.prop = object_probabilities[i];
        result.cls_id = class_ids[candidate_index];
        ++od_results->count;
    }

    return 0;
}
