#ifndef _RKNN_YOLOV8_POSE_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_POSE_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"

struct YoloConfig {
    // Storage capacities remain compile-time constants so the public result
    // structs stay simple and allocation-free.
    static constexpr int keypoint_capacity = 17;
    static constexpr int result_capacity = 128;
    static constexpr int keypoint_value_count = 3; // x, y, confidence
    static constexpr int box_record_size = 5;      // x, y, w, h, anchor

    int class_count = 1;
    int keypoint_count = 17;
    int keypoint_dimensions = 3;
    int dfl_bins = 16;
    int detection_head_count = 3;
    int keypoint_output_index = 3;
    int anchor_count = 8400;
    int max_results = result_capacity;
    float nms_threshold = 0.4f;
    float box_threshold = 0.5f;
    bool debug_output = false;
    std::string labels_path = "./model/yolov8_pose_labels_list.txt";
};

// Configure this before inference. The returned object has the official
// YOLOv8-pose defaults until changed by the application.
YoloConfig& yolo_config();

struct rknn_app_context_t;

typedef struct {
    image_rect_t box;
    float keypoints[YoloConfig::keypoint_capacity]
                   [YoloConfig::keypoint_value_count];
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[YoloConfig::result_capacity];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

void deinitPostProcess();
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
