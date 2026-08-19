#ifndef DESKTOP_DEBUG
#include "rknn_api.h"
#endif
#include "rcj_vision.hpp"
#include <opencv2/core.hpp>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

namespace {
    struct LetterboxResult {
        cv::Mat rgb;
        float scale;
        int pad_x;
        int pad_y;
    };

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

    struct GateSegmentCandidate {
        int color;
        cv::Point2f left;
        cv::Point2f right;
        float confidence; 
    };

    ww::vision::Point RestorePointFromLetterbox(
        const cv::Mat& frame,
        const LetterboxResult& prep,
        cv::Point2f point) {
        
        point.x = (point.x - prep.pad_x) / prep.scale;
        point.y = (point.y - prep.pad_y) / prep.scale;

        point.x = std::clamp(point.x, 0.0f, float(frame.cols - 1));
        point.y = std::clamp(point.y, 0.0f, float(frame.rows - 1));

        return ww::vision::PointFromImage(point);
    }
}

namespace ww {
namespace vision {

#define GRACEFUL_ASSERT(val, msg, err_code) \
if (val) { RK_LOGE(msg); return err_code; }

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
    static constexpr int kExpectedWidth = 320;
    static constexpr int kExpectedHeight = 320;
    static constexpr int kOutputChannels = 12;

    static constexpr float kDetectionThreshold = 0;
    static constexpr float kKeypointThreshold = 0;

    enum ModelChannel {
        kBoxCenterX = 0,
        kBoxCenterY = 1,
        kBoxWidth = 2,
        kBoxHeight = 3,
        kYellowScore = 4,
        kBlueScore = 5,
        kLeftPointX = 6,
        kLeftPointY = 7,
        kLeftPointConfidence = 8,
        kRightPointX = 9,
        kRightPointY = 10,
        kRightPointConfidence = 11
    };

    class GateSegmentDetector::Impl {
    public:
        rknn_context context = 0;
        rknn_tensor_attr input_attr{};
        rknn_tensor_attr output_attr{};

        int input_width = 0;
        int input_height = 0;

        const char* kModelPath = "/root/best-int8.rknn";

        bool ValidateModel() {
            RK_LOGI("Validate model...");
            rknn_input_output_num io_num{};

            GRACEFUL_ASSERT_EQ(
                (rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num))),
                RKNN_SUCC,
                "RKNN_QUERY_IN_OUT_NUM failed", false);

            GRACEFUL_ASSERT_EQ(
                (io_num.n_input), 1,
                "wrong io_num.n_input", false);

            GRACEFUL_ASSERT_EQ(
                (io_num.n_output), 1,
                "wrong io_num.n_output", false);

            input_attr.index = 0;
            GRACEFUL_ASSERT_EQ(
                (rknn_query(
                    context, RKNN_QUERY_INPUT_ATTR,
                    &input_attr, sizeof(input_attr)
                )),
                RKNN_SUCC,
                "RKNN_QUERY_INPUT_ATTR failed", false);

            output_attr.index = 0;
            GRACEFUL_ASSERT_EQ(
                (rknn_query(
                    context, RKNN_QUERY_OUTPUT_ATTR,
                    &output_attr, sizeof(output_attr)
                )),
                RKNN_SUCC,
                "RKNN_QUERY_OUTPUT_ATTR failed", false);

            if (input_attr.fmt == RKNN_TENSOR_NCHW) {
                input_height = input_attr.dims[2];
                input_width = input_attr.dims[3];
            } else {
                input_height = input_attr.dims[1];
                input_width = input_attr.dims[2];
            }

            GRACEFUL_ASSERT_EQ(
                input_width, kExpectedWidth,
                "Wrong model width", false);

            GRACEFUL_ASSERT_EQ(
                input_height, kExpectedHeight,
                "Wrong model height", false);

            GRACEFUL_ASSERT_EQ(
                (output_attr.n_elems % kOutputChannels), 0,
                "n_elems \% channels", false);

            RK_LOGI("Model is correct!");
            return true;
        }

        Impl() {
            RK_LOGI("Initialize model...");
            GRACEFUL_ASSERT_EQ(
                (rknn_init(&context, const_cast<char*>(kModelPath), 0, 0, nullptr)),
                RKNN_SUCC,
                "rknn_init", );
                
            if (!ValidateModel()) {
                rknn_destroy(context);
                context = 0;
                return;
            }
        }

        template <typename ValueGetter>
        std::optional<GateSegmentCandidate> DecodeGate(int i, ValueGetter&& value) {
            float yellow_score = value(kYellowScore, i);
            float blue_score = value(kBlueScore, i);

            int color = blue_score > yellow_score ? 1 : 0;
            float score = std::max(yellow_score, blue_score);

            if (score < kDetectionThreshold ||
                value(kLeftPointConfidence, i) < kKeypointThreshold ||
                value(kRightPointConfidence, i) < kKeypointThreshold) {
                return std::nullopt;
            }

            return GateSegmentCandidate{
                color,
                cv::Point2f{value(kLeftPointX, i), value(kLeftPointY, i)},
                cv::Point2f{value(kRightPointX, i), value(kRightPointY, i)},
                std::min(score,
                    std::min(
                        value(kLeftPointConfidence, i),
                        value(kRightPointConfidence, i)))
            };
        }

        FieldObjects ProcessGateCandidates(
            const cv::Mat& frame,
            const LetterboxResult& prep,
            std::vector<GateSegmentCandidate>& candidates) {
            
            auto middle = candidates.begin();
            
            std::partition(candidates.begin(), candidates.end(),
                [](const GateSegmentCandidate& cand){ return cand.color; });
            
            auto comp_candidates =
                [](const GateSegmentCandidate& lhs, const GateSegmentCandidate& rhs) {
                    return lhs.confidence < rhs.confidence;
                };

            std::sort(candidates.begin(), middle, comp_candidates);
            std::sort(middle, candidates.end(), comp_candidates);

            FieldObjects fo;
            auto last_yellow = (
                middle - candidates.begin() < max_one_color_gates_
                ? candidates.begin() + max_one_color_gates_
                : middle);

            auto last_blue = (
                candidates.end() - middle < max_one_color_gates_
                ? middle + max_one_color_gates_
                : candidates.end());
            
            for (auto iter = candidates.begin(); iter != last_yellow; ++iter) {
                fo.yellow_gates.emplace_back(Segment(
                    RestorePointFromLetterbox(frame, prep, iter->left),
                    RestorePointFromLetterbox(frame, prep, iter->right)
                ));
            }
            for (auto iter = middle; iter != last_blue; ++iter) {
                fo.blue_gates.emplace_back(Segment(
                    RestorePointFromLetterbox(frame, prep, iter->left),
                    RestorePointFromLetterbox(frame, prep, iter->right)
                ));
            }
            return fo;
        }

        ~Impl() {
            if (context != 0) {
                rknn_destroy(context);
            }
        }
    };

    GateSegmentDetector::GateSegmentDetector()
        : pimpl_(std::make_unique<Impl>()) {}

    auto GateSegmentDetector::Detect(cv::Mat& frame)
        -> std::optional<FieldObjects> {
            
        GRACEFUL_ASSERT_NEQ(
            (pimpl_->context),
            0,
            "rknn hasn't initialized", (std::nullopt));

        auto prep = Letterbox(
            frame, pimpl_->input_width, pimpl_->input_height
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
            (rknn_inputs_set(pimpl_->context, 1, &input) == RKNN_SUCC),
            "rknn_inputs_set failed", (std::nullopt));
        
        GRACEFUL_ASSERT(
            (rknn_run(pimpl_->context, nullptr) == RKNN_SUCC),
            "rknn_run failed", (std::nullopt));

        rknn_output output{};
        output.index = 0;
        output.want_float = 1;
        output.is_prealloc = 0;
        
        GRACEFUL_ASSERT(
            (rknn_outputs_get(pimpl_->context, 1, &output, nullptr) == RKNN_SUCC),
            "rknn_outputs_get failed", (std::nullopt));

        const float* prediction = static_cast<const float*>(output.buf);
        const int count = pimpl_->output_attr.n_elems / kOutputChannels;

        auto value = [prediction, count](int channel, int index) {
            return prediction[channel * count + index];
        };

        std::vector<GateSegmentCandidate> candidates;
        candidates.reserve(count);
        for (int i = 0; i < count; ++i) {
            auto gate_opt = pimpl_->DecodeGate(i, value);
            if (gate_opt) {
                candidates.push_back(std::move(*gate_opt));
            }
        }
        
        rknn_outputs_release(pimpl_->context, 1, &output);

        FieldObjects fo = pimpl_->ProcessGateCandidates(
            frame, prep, candidates);
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
