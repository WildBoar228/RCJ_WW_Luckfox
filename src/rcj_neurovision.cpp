#ifndef DESKTOP_DEBUG
#include "rknn_api.h"
#include "postprocess.h"
#include "yolov8-pose.h"
#endif
#include "rcj_vision.hpp"
#include "rcj_neurovision.hpp"
#include <opencv2/core.hpp>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

namespace ww {
namespace vision {

namespace {

    ww::vision::Point PointFromDetection(
        const cv::Mat& frame,
        cv::Point2f point) {

        point.x = std::clamp(point.x, 0.0f, float(frame.cols - 1));
        point.y = std::clamp(point.y, 0.0f, float(frame.rows - 1));

        return ww::vision::PointFromImage(point);
    }
} // namespace

#define GRACEFUL_ASSERT(val, msg, err_code) \
if (!(val)) { RK_LOGE(msg); return err_code; }

#define GRACEFUL_ASSERT_EQ(exp, actual, msg, err_code)  \
if (exp != actual) {                                    \
    RK_LOGE(msg);                                       \
    RK_LOGW("Expected %d, got %d", exp, actual);        \
    return err_code;                                    \
}

#define GRACEFUL_ASSERT_NEQ(exp, actual, msg, err_code) \
if (exp == actual) {                                    \
    RK_LOGE(msg);                                       \
    RK_LOGW("Expected %d, got %d", exp, actual);        \
    return err_code;                                    \
}

#ifndef DESKTOP_DEBUG

    FieldObjects ProcessGateCandidates(
        const cv::Mat& frame,
        std::vector<GateSegmentCandidate>& candidates,
        int max_one_color_gates
    ) {
        auto middle = std::partition(
            candidates.begin(), candidates.end(),
            [](const GateSegmentCandidate& candidate) {
                return candidate.color == 0;
            }
        );

        auto comp_candidates =
            [](const GateSegmentCandidate& lhs, const GateSegmentCandidate& rhs) {
                return lhs.confidence > rhs.confidence;
            };

        std::sort(candidates.begin(), middle, comp_candidates);
        std::sort(middle, candidates.end(), comp_candidates);

        FieldObjects fo;
        auto last_yellow = candidates.begin() + std::min(
            static_cast<int>(middle - candidates.begin()),
            max_one_color_gates);

        auto last_blue = middle + std::min(
            static_cast<int>(candidates.end() - middle),
            max_one_color_gates);

        std::cout << "yellow cnt:  " << (last_yellow - candidates.begin()) << "\n";
        std::cout << "blue cnt:  " << (last_blue - middle) << "\n";

        for (auto iter = candidates.begin(); iter != last_yellow; ++iter) {
            fo.yellow_gates.emplace_back(Segment(
                PointFromDetection(frame, iter->left),
                PointFromDetection(frame, iter->right)
            ));
        }
        for (auto iter = middle; iter != last_blue; ++iter) {
            fo.blue_gates.emplace_back(Segment(
                PointFromDetection(frame, iter->left),
                PointFromDetection(frame, iter->right)
            ));
        }
        return fo;
    }

    GateSegmentDetector::GateSegmentDetector(char* model_path)
        : is_initialized(false) {
        RK_LOGI("Initialize model...");

        GRACEFUL_ASSERT_EQ(
            (init_yolov8_pose_model(model_path, &app_ctx)),
            0,
            "init_yolov8_pose_model", );

        is_initialized = true;
    }

    auto GateSegmentDetector::Detect(cv::Mat& frame, int dma_fd)
        -> std::optional<FieldObjects> {

        RK_LOGI("Detect...\n");

        if (!is_initialized) {
            RK_LOGE("GateSegmentDetector isn't initialized!");
            return std::nullopt;
        }

        object_detect_result_list od_results;

        // cv::Mat rgb;
        // cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        src_image = {};
        src_image.width = frame.cols;
        src_image.height = frame.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = frame.data;
        src_image.size = static_cast<int>(frame.step * frame.rows);
        src_image.fd = -1;

        int ret = inference_yolov8_pose_model(&app_ctx, &src_image, &od_results);
        if (ret != 0) {
            RK_LOGE("inference_yolov8_pose_model failed: %d", ret);
            return std::nullopt;
        }

        RK_LOGI("Inference done\n");

        std::vector<GateSegmentCandidate> candidates;
        candidates.reserve(od_results.count);
        for (int i = 0; i < od_results.count; ++i) {
            const object_detect_result& result = od_results.results[i];
            if (result.cls_id < 0 || result.cls_id >= kClassNum) {
                RK_LOGW("reject1:  class %d", result.cls_id);
                continue;
            }

            const float left_confidence = result.keypoints[0][2];
            const float right_confidence = result.keypoints[1][2];
            if (left_confidence < kKeypointThreshold ||
                right_confidence < kKeypointThreshold) {
                RK_LOGW("reject2 (confidence):  left %f, right %f, threshold %f",
                    left_confidence, right_confidence, kKeypointThreshold);
                continue;
            }

            RK_LOGW("detect class %d", result.cls_id);

            candidates.push_back(GateSegmentCandidate{
                result.cls_id,
                cv::Point2f{
                    result.keypoints[0][0], result.keypoints[0][1]
                },
                cv::Point2f{
                    result.keypoints[1][0], result.keypoints[1][1]
                },
                std::min(
                    result.prop,
                    std::min(left_confidence, right_confidence)
                )
            });
        }

        FieldObjects fo = ProcessGateCandidates(
            frame, candidates, max_one_color_gates_);

        std::cout << "yellow cnt:  " << fo.yellow_gates.size() << "\n";
        std::cout << "blue cnt:  " << fo.blue_gates.size() << "\n";

        RK_LOGI("Detecting ended\n");
        return fo;
    }

    void GateSegmentDetector::DrawResult(cv::Mat& frame, const FieldObjects& field) {
        std::cout << "yellow cnt:  " << field.yellow_gates.size() << "\n";
        std::cout << "blue cnt:  " << field.blue_gates.size() << "\n";

        for (const Segment& ygate : field.yellow_gates) {
            RK_LOGW("yellow gate: (%f;%f) (%f;%f)",
                PointToImage(ygate.Begin()).x,
                PointToImage(ygate.Begin()).y,
                PointToImage(ygate.End()).x,
                PointToImage(ygate.End()).y
            );
            cv::line(
                frame,
                PointToImage(ygate.Begin()),
                PointToImage(ygate.End()),
                vision_cfg.default_thr_colors[0], 2
            );
        }
        for (const Segment& bgate : field.blue_gates) {
            RK_LOGW("blue gate: (%f;%f) (%f;%f)",
                PointToImage(bgate.Begin()).x,
                PointToImage(bgate.Begin()).y,
                PointToImage(bgate.End()).x,
                PointToImage(bgate.End()).y
            );
            cv::line(
                frame,
                PointToImage(bgate.Begin()),
                PointToImage(bgate.End()),
                vision_cfg.default_thr_colors[1], 2
            );
        }
    }

    GateSegmentDetector::~GateSegmentDetector() {
        release_yolov8_pose_model(&app_ctx);
        is_initialized = false;
    }
#else
#endif

} // namespace vision
} // namespace ww
