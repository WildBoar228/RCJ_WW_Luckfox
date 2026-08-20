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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov8-pose.h"
#include "common.h"
// #include "image_utils.h"

#include <sys/time.h>

#define YOLO_DEBUG_PRINT(...)                         \
    do {                                              \
        if (yolo_config().debug_output) {             \
            printf(__VA_ARGS__);                      \
        }                                             \
    } while (0)

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    YOLO_DEBUG_PRINT(
        "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
        "zp=%d, scale=%f, w_stride=%d, size_with_stride=%d\n",
        attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
        attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale,
        attr->w_stride, attr->size_with_stride);
}

int init_yolov8_pose_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    if (model_path == NULL || app_ctx == NULL) {
        return -1;
    }

    memset(app_ctx, 0, sizeof(*app_ctx));

    int ret = RKNN_ERR_FAIL;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret != RKNN_SUCC || ctx == 0)
    {
        printf("rknn_init fail! ret=%d, ctx=%u\n", ret, (unsigned int)ctx);
        return -1;
    }
    app_ctx->rknn_ctx = ctx;
    const auto fail_init = [app_ctx]() {
        release_yolov8_pose_model(app_ctx);
        return -1;
    };

    rknn_sdk_version version{};
    ret = rknn_query(
        ctx,
        RKNN_QUERY_SDK_VERSION,
        &version,
        sizeof(version));

    if (ret == RKNN_SUCC) {
        YOLO_DEBUG_PRINT("RKNN API: %s\n", version.api_version);
        YOLO_DEBUG_PRINT("RKNN driver: %s\n", version.drv_version);
    } else {
        printf("FAILED rknn_query: %d\n", ret);
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return fail_init();
    }
    YOLO_DEBUG_PRINT("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    const YoloConfig& config = yolo_config();
    if (io_num.n_input != 1 ||
        io_num.n_output != static_cast<uint32_t>(config.detection_head_count + 1) ||
        config.keypoint_output_index >= static_cast<int>(io_num.n_output)) {
        printf("unexpected model I/O count: inputs=%u outputs=%u\n",
               io_num.n_input, io_num.n_output);
        return fail_init();
    }
    app_ctx->io_num = io_num;

    app_ctx->input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    app_ctx->input_mems = (rknn_tensor_mem **)calloc(io_num.n_input, sizeof(rknn_tensor_mem *));
    app_ctx->output_mems = (rknn_tensor_mem **)calloc(io_num.n_output, sizeof(rknn_tensor_mem *));
    if (app_ctx->input_attrs == NULL || app_ctx->output_attrs == NULL ||
        app_ctx->input_mems == NULL || app_ctx->output_mems == NULL) {
        printf("allocating RKNN context metadata failed\n");
        return fail_init();
    }

    // RV1106 uses pre-bound native input memory.
    YOLO_DEBUG_PRINT("native input tensors:\n");
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        rknn_tensor_attr& attr = app_ctx->input_attrs[i];
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC)
        {
            printf("RKNN_QUERY_NATIVE_INPUT_ATTR failed! ret=%d\n", ret);
            return fail_init();
        }
        dump_tensor_attr(&attr);
    }

    // Request unpacked NHWC outputs so CPU postprocessing can index them.
    YOLO_DEBUG_PRINT("native NHWC output tensors:\n");
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC)
        {
            printf("RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR failed! ret=%d\n", ret);
            return fail_init();
        }
        dump_tensor_attr(&attr);
    }

    app_ctx->is_quant =
        app_ctx->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        app_ctx->output_attrs[0].type == RKNN_TENSOR_INT8;

    const rknn_tensor_attr& input_attr = app_ctx->input_attrs[0];
    if (input_attr.fmt == RKNN_TENSOR_NCHW)
    {
        YOLO_DEBUG_PRINT("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attr.dims[1];
        app_ctx->model_height = input_attr.dims[2];
        app_ctx->model_width = input_attr.dims[3];
    }
    else
    {
        YOLO_DEBUG_PRINT("model is NHWC input fmt\n");
        app_ctx->model_height = input_attr.dims[1];
        app_ctx->model_width = input_attr.dims[2];
        app_ctx->model_channel = input_attr.dims[3];
    }
    YOLO_DEBUG_PRINT("model input height=%d, width=%d, channel=%d\n",
                     app_ctx->model_height, app_ctx->model_width,
                     app_ctx->model_channel);

    // Ask RKNN to fuse UINT8 RGB normalization/quantization into the input.
    rknn_tensor_attr& bound_input_attr = app_ctx->input_attrs[0];
    bound_input_attr.type = RKNN_TENSOR_UINT8;
    bound_input_attr.fmt = RKNN_TENSOR_NHWC;
    bound_input_attr.pass_through = 0;

    uint32_t input_size = bound_input_attr.size_with_stride != 0
        ? bound_input_attr.size_with_stride
        : bound_input_attr.size;
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_size);
    if (app_ctx->input_mems[0] == NULL) {
        printf("rknn_create_mem for input failed, size=%u\n", input_size);
        return fail_init();
    }
    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &bound_input_attr);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_io_mem for input failed! ret=%d\n", ret);
        return fail_init();
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        rknn_tensor_attr& attr = app_ctx->output_attrs[i];
        uint32_t output_size = attr.size_with_stride != 0
            ? attr.size_with_stride
            : attr.size;
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_size);
        if (app_ctx->output_mems[i] == NULL) {
            printf("rknn_create_mem for output %u failed, size=%u\n", i, output_size);
            return fail_init();
        }
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &attr);
        if (ret != RKNN_SUCC) {
            printf("rknn_set_io_mem for output %u failed! ret=%d\n", i, ret);
            return fail_init();
        }
    }

    return 0;
}

int release_yolov8_pose_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx == NULL) {
        return 0;
    }

    if (app_ctx->input_mems != NULL) {
        for (uint32_t i = 0; i < app_ctx->io_num.n_input; ++i) {
            if (app_ctx->input_mems[i] != NULL && app_ctx->rknn_ctx != 0) {
                rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
            }
        }
        free(app_ctx->input_mems);
        app_ctx->input_mems = NULL;
    }
    if (app_ctx->output_mems != NULL) {
        for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
            if (app_ctx->output_mems[i] != NULL && app_ctx->rknn_ctx != 0) {
                rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
            }
        }
        free(app_ctx->output_mems);
        app_ctx->output_mems = NULL;
    }
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}


int inference_yolov8_pose_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    int ret = RKNN_ERR_FAIL;
    image_buffer_t dst_img{};
    letterbox_t letter_box{};
    const YoloConfig& config = yolo_config();
    set_image_utils_debug(config.debug_output ? 1 : 0);
    const float nms_threshold = config.nms_threshold;
    const float box_conf_threshold = config.box_threshold;
    int bg_color = 114;

    if (app_ctx == NULL || img == NULL || od_results == NULL ||
        app_ctx->rknn_ctx == 0 || app_ctx->input_mems == NULL ||
        app_ctx->input_mems[0] == NULL || app_ctx->output_mems == NULL)
    {
        return -1;
    }

    int start_us,end_us;

    memset(od_results, 0x00, sizeof(*od_results));
    // Pre Process

    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)app_ctx->input_mems[0]->virt_addr;
    dst_img.fd = app_ctx->input_mems[0]->fd;
    if (dst_img.virt_addr == NULL || dst_img.size > (int)app_ctx->input_mems[0]->size)
    {
        printf("RKNN input buffer is invalid: required=%d available=%u\n",
               dst_img.size, app_ctx->input_mems[0]->size);
        return -1;
    }

    // letterbox
    start_us = getCurrentTimeUs();
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    end_us = getCurrentTimeUs() - start_us;
    YOLO_DEBUG_PRINT("convert_image_with_letterbox time=%.2fms, FPS = %.2f\n",end_us / 1000.f,
                     1000.f * 1000.f / end_us);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return ret;
    }

    // Run
    YOLO_DEBUG_PRINT("rknn_run\n");
    start_us = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    end_us = getCurrentTimeUs() - start_us;
    YOLO_DEBUG_PRINT("rknn_run time=%.2fms, FPS = %.2f\n",end_us / 1000.f,
                     1000.f * 1000.f / end_us);

    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }

    // Post Process
    start_us = getCurrentTimeUs();
    ret = post_process(app_ctx, app_ctx->output_mems, &letter_box,
                       box_conf_threshold, nms_threshold, od_results);
    end_us = getCurrentTimeUs() - start_us;
    YOLO_DEBUG_PRINT("post_process time=%.2fms, FPS = %.2f\n",end_us / 1000.f,
                     1000.f * 1000.f / end_us);
    if (ret != 0) {
        printf("post_process failed! ret=%d\n", ret);
    }

    return ret;
}
