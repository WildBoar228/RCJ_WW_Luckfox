#ifndef _RKNN_YOLOV8_POSE_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_POSE_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "rcj_neurovision.hpp"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 8
#define OBJ_CLASS_NUM ww::vision::kClassNum
#define NMS_THRESH 0.4
#define BOX_THRESH ww::vision::kDetectionThreshold
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

// class rknn_app_context_t;

typedef struct {
    float keypoints
        [ww::vision::kKeypointNum]
        [ww::vision::kKeypointDimensions]; // x, y, confidence
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int post_process(
    ww::vision::ModelHandler*,
    void* outputs,
    ww::vision::LetterboxResult*,
    float conf_threshold,
    float nms_threshold,
    object_detect_result_list* od_results);

#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
