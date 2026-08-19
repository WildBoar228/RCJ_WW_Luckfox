#ifndef DESKTOP_DEBUG
#include "rknn_api.h"
#endif
#include "rcj_vision.hpp"
#include "rcj_neurovision.hpp"
#include "postprocess.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

namespace ww {
namespace vision {

namespace {

    LetterboxResult Letterbox(const cv::Mat& frame, int width, int height) {
        float scale = std::min(
            static_cast<float>(width) / frame.cols,
            static_cast<float>(height) / frame.rows
        );

        int resized_width =
            static_cast<int>(std::round(frame.cols * scale));
        int resized_height =
            static_cast<int>(std::round(frame.rows * scale));

        int pad_x = (width - resized_width) / 2;
        int pad_y = (height - resized_height) / 2;

        cv::Mat input(
            height, width, CV_8UC3,
            cv::Scalar(114, 114, 114)
        );

        cv::Mat resized;
        cv::resize(
            frame, resized,
            cv::Size(resized_width, resized_height),
            0, 0, cv::INTER_LINEAR
        );

        resized.copyTo(input(
            cv::Rect(pad_x, pad_y, resized_width, resized_height)
        ));

        cv::Mat rgb;
        cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);

        return {std::move(rgb), scale, pad_x, pad_y};
    }

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

    ModelHandler::ModelHandler(char* model_path) {
        RK_LOGI("Initialize model...");
        GRACEFUL_ASSERT_EQ(
            (rknn_init(&context, model_path, 0, 0, nullptr)),
            RKNN_SUCC,
            "rknn_init", );

        if (!ValidateModel()) {
            rknn_destroy(context);
            context = 0;
            return;
        }
    }

    ModelHandler::~ModelHandler() {
        if (context != 0) {
            rknn_destroy(context);
        }
    }

    bool ModelHandler::ValidateModel() {
        RK_LOGI("Validate model...");

        GRACEFUL_ASSERT_EQ(
            (rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num))),
            RKNN_SUCC,
            "RKNN_QUERY_IN_OUT_NUM failed", false);

        GRACEFUL_ASSERT_EQ(
            (io_num.n_input), 1,
            "wrong io_num.n_input", false);

        GRACEFUL_ASSERT_EQ(
            (io_num.n_output), kModelOutputNum,
            "wrong io_num.n_output", false);

        input_attrs[0].index = 0;
        GRACEFUL_ASSERT_EQ(
            (rknn_query(
                context, RKNN_QUERY_INPUT_ATTR,
                &input_attrs[0], sizeof(input_attrs[0])
            )),
            RKNN_SUCC,
            "RKNN_QUERY_INPUT_ATTR failed", false);

        for (int i = 0; i < kModelOutputNum; ++i) {
            output_attrs[i].index = i;
            GRACEFUL_ASSERT_EQ(
                (rknn_query(
                    context, RKNN_QUERY_OUTPUT_ATTR,
                    &output_attrs[i], sizeof(output_attrs[i])
                )),
                RKNN_SUCC,
                "RKNN_QUERY_OUTPUT_ATTR failed", false);
        }

        const auto& input_attr = input_attrs[0];
        GRACEFUL_ASSERT_EQ(
            input_attr.n_dims, 4,
            "Input must have four dimensions", false);
        if (input_attr.fmt == RKNN_TENSOR_NCHW) {
            input_height = input_attr.dims[2];
            input_width = input_attr.dims[3];
            GRACEFUL_ASSERT_EQ(
                input_attr.dims[1], 3,
                "Wrong model input channels", false);
        } else {
            input_height = input_attr.dims[1];
            input_width = input_attr.dims[2];
            GRACEFUL_ASSERT_EQ(
                input_attr.dims[3], 3,
                "Wrong model input channels", false);
        }

        GRACEFUL_ASSERT_EQ(
            input_width, kExpectedWidth,
            "Wrong model width", false);

        GRACEFUL_ASSERT_EQ(
            input_height, kExpectedHeight,
            "Wrong model height", false);

        constexpr std::array<int, kDetectionHeadNum> grid_sizes{40, 20, 10};
        for (int i = 0; i < kDetectionHeadNum; ++i) {
            const auto& attr = output_attrs[i];
            const int grid = grid_sizes[i];

            GRACEFUL_ASSERT_EQ(
                attr.n_dims, 4,
                "Detection output must have four dimensions", false);
            GRACEFUL_ASSERT_EQ(
                attr.fmt, RKNN_TENSOR_NCHW,
                "Detection output must use NCHW", false);
            GRACEFUL_ASSERT_EQ(
                attr.dims[0], 1,
                "Wrong detection output batch", false);
            GRACEFUL_ASSERT_EQ(
                attr.dims[1], kDetectionOutputChannels,
                "Wrong detection output channels", false);
            GRACEFUL_ASSERT_EQ(
                static_cast<int>(attr.dims[2]), grid,
                "Wrong detection output height", false);
            GRACEFUL_ASSERT_EQ(
                static_cast<int>(attr.dims[3]), grid,
                "Wrong detection output width", false);
            GRACEFUL_ASSERT(
                attr.type == RKNN_TENSOR_INT8 ||
                attr.type == RKNN_TENSOR_UINT8 ||
                attr.type == RKNN_TENSOR_FLOAT16 ||
                attr.type == RKNN_TENSOR_FLOAT32,
                "Unsupported detection output type", false);
        }

        const auto& keypoint_attr = output_attrs[3];
        GRACEFUL_ASSERT_EQ(
            keypoint_attr.n_dims, 4,
            "Keypoint output must have four dimensions", false);
        GRACEFUL_ASSERT_EQ(
            keypoint_attr.dims[0], 1,
            "Wrong keypoint output batch", false);
        GRACEFUL_ASSERT_EQ(
            keypoint_attr.dims[1], kKeypointNum,
            "Wrong number of keypoints", false);
        GRACEFUL_ASSERT_EQ(
            keypoint_attr.dims[2], kKeypointDimensions,
            "Wrong keypoint dimensions", false);
        GRACEFUL_ASSERT_EQ(
            keypoint_attr.dims[3], kTotalAnchors,
            "Wrong keypoint anchor count", false);
        GRACEFUL_ASSERT(
            keypoint_attr.type == RKNN_TENSOR_INT8 ||
            keypoint_attr.type == RKNN_TENSOR_UINT8 ||
            keypoint_attr.type == RKNN_TENSOR_FLOAT16 ||
            keypoint_attr.type == RKNN_TENSOR_FLOAT32,
            "Unsupported keypoint output type", false);

        is_quant =
            output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
            (output_attrs[0].type == RKNN_TENSOR_INT8 ||
             output_attrs[0].type == RKNN_TENSOR_UINT8);

        for (int i = 1; i < kDetectionHeadNum; ++i) {
            const bool output_is_quant =
                output_attrs[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                (output_attrs[i].type == RKNN_TENSOR_INT8 ||
                 output_attrs[i].type == RKNN_TENSOR_UINT8);
            GRACEFUL_ASSERT_EQ(
                output_is_quant, is_quant,
                "Detection heads use inconsistent quantization", false);
        }

        RK_LOGI("Model is correct!");
        return true;
    }

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
        : model_(model_path) {}

    auto GateSegmentDetector::Detect(cv::Mat& frame)
        -> std::optional<FieldObjects> {
            
        GRACEFUL_ASSERT_NEQ(
            (model_.context),
            0,
            "rknn hasn't initialized", (std::nullopt));
        GRACEFUL_ASSERT(
            !frame.empty(),
            "input frame is empty", (std::nullopt));

        auto prep = Letterbox(
            frame, model_.input_width, model_.input_height
        );

        rknn_input input{};
        input.index = 0;
        input.buf = prep.rgb.data;
        input.size = static_cast<uint32_t>(
            prep.rgb.total() * prep.rgb.elemSize()
        );
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.pass_through = 0;

        GRACEFUL_ASSERT(
            (rknn_inputs_set(model_.context, 1, &input) == RKNN_SUCC),
            "rknn_inputs_set failed", (std::nullopt));
        
        GRACEFUL_ASSERT(
            (rknn_run(model_.context, nullptr) == RKNN_SUCC),
            "rknn_run failed", (std::nullopt));

        std::array<rknn_output, kModelOutputNum> outputs{};
        for (int i = 0; i < kModelOutputNum; ++i) {
            outputs[i].index = i;
            // Match Rockchip's pose example: preserve native INT8/FP16
            // outputs for a quantized model, otherwise request FP32.
            outputs[i].want_float = !model_.is_quant;
            outputs[i].is_prealloc = 0;
        }

        GRACEFUL_ASSERT(
            (rknn_outputs_get(
                model_.context,
                kModelOutputNum,
                outputs.data(),
                nullptr
            ) == RKNN_SUCC),
            "rknn_outputs_get failed", (std::nullopt));

        object_detect_result_list results{};
        const int postprocess_result = post_process(
            &model_,
            outputs.data(),
            &prep,
            BOX_THRESH,
            NMS_THRESH,
            &results
        );
        const int release_result = rknn_outputs_release(
            model_.context, kModelOutputNum, outputs.data());

        GRACEFUL_ASSERT_EQ(
            postprocess_result, 0,
            "pose postprocessing failed", (std::nullopt));
        GRACEFUL_ASSERT_EQ(
            release_result, RKNN_SUCC,
            "rknn_outputs_release failed", (std::nullopt));

        std::vector<GateSegmentCandidate> candidates;
        candidates.reserve(results.count);
        for (int i = 0; i < results.count; ++i) {
            const auto& result = results.results[i];
            if (result.cls_id < 0 || result.cls_id >= kClassNum) {
                continue;
            }

            const float left_confidence = result.keypoints[0][2];
            const float right_confidence = result.keypoints[1][2];
            if (left_confidence < kKeypointThreshold ||
                right_confidence < kKeypointThreshold) {
                continue;
            }

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
        return fo;
    }

    void GateSegmentDetector::DrawResult(cv::Mat& frame, const FieldObjects& field) {
        for (const Segment& ygate : field.yellow_gates) {
            cv::line(
                frame,
                PointToImage(ygate.Begin()),
                PointToImage(ygate.End()),
                vision_cfg.default_thr_colors[0], 2
            );
        }
        for (const Segment& bgate : field.blue_gates) {
            cv::line(
                frame,
                PointToImage(bgate.Begin()),
                PointToImage(bgate.End()),
                vision_cfg.default_thr_colors[1], 2
            );
        }
    }

    GateSegmentDetector::~GateSegmentDetector() = default;
#else
#endif

} // namespace vision
} // namespace ww
