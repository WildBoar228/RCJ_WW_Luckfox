#ifndef _RCJ_WW_LUCKFOX_RCJ_NEUROVISION_HPP_
#define _RCJ_WW_LUCKFOX_RCJ_NEUROVISION_HPP_

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifndef DESKTOP_DEBUG
#include "rk_debug.h"
#include "rk_defines.h"
#include "rk_mpi_adec.h"
#include "rk_mpi_aenc.h"
#include "rk_mpi_ai.h"
#include "rk_mpi_ao.h"
#include "rk_mpi_avs.h"
#include "rk_mpi_cal.h"
#include "rk_mpi_ivs.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_rgn.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_tde.h"
#include "rk_mpi_vdec.h"
#include "rk_mpi_venc.h"
#include "rk_mpi_vi.h"
#include "rk_mpi_vo.h"
#include "rk_mpi_vpss.h"

#include "rknn_api.h"
#include "yolov8-pose.h"
#endif

#include "rtsp_demo.h"
#include "sample_comm.h"

#include "geometry.hpp"

namespace ww {
namespace vision {

    static constexpr int kClassNum = 2;
    static constexpr int kKeypointNum = 2;
    static constexpr int kKeypointDimensions = 3;
    static constexpr int kDflBins = 16;
    static constexpr int kDetectionHeadNum = 3;
    static constexpr int kModelOutputNum = 4;

    static constexpr int kExpectedWidth = 320;
    static constexpr int kExpectedHeight = 320;
    static constexpr int kDetectionOutputChannels =
        4 * kDflBins + kClassNum;
    static constexpr int kTotalAnchors =
        40 * 40 + 20 * 20 + 10 * 10;

    static constexpr float kDetectionThreshold = 0.05; //0.5f;
    static constexpr float kKeypointThreshold = 0.05; //0.25f;

    struct FieldObjects {
        std::vector<Segment> yellow_gates;
        std::vector<Segment> blue_gates;
    };

    struct GateSegmentCandidate {
        int color;
        cv::Point2f left;
        cv::Point2f right;
        float confidence;
    };

    class GateSegmentDetector {
        rknn_app_context_t app_ctx{};
        image_buffer_t src_image{};
        static constexpr int max_one_color_gates_ = 2;
        bool is_initialized = false;

    public:
        GateSegmentDetector(char* model_path);
        
        std::optional<FieldObjects> Detect(cv::Mat&, int dma_fd);
        static void DrawResult(cv::Mat&, const FieldObjects&);
        ~GateSegmentDetector();
    };

    FieldObjects ProcessGateCandidates(
        const cv::Mat&,
        std::vector<GateSegmentCandidate>&,
        int max_one_color_gates
    );

} // namespace vision
} // namespace ww

#endif
