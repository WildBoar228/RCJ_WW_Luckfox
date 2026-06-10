#ifndef _RCJ_WW_LUCKFOX_RCJ_VISION_HPP_
#define _RCJ_WW_LUCKFOX_RCJ_VISION_HPP_

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifdef DESKTOP_DEBUG
#include <opencv2/videoio.hpp>
#else 

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

#include "rtsp_demo.h"
#include "sample_comm.h"

#endif

#include "geometry.hpp"

namespace ww_vision {

    using ColorLab = cv::Scalar;

    Point PointFromImage(const cv::Point2f&);
    cv::Point2f PointToImage(const Point&);

    struct ColorThreshold {
        ColorLab lower;
        ColorLab upper;
    };

    struct BlobInfo {
        Deg left_angle = 360_deg;
        Deg right_angle = 360_deg;
        Deg center_angle = 360_deg;
        Deg width = 0_deg;
        Deg clos_angle = 360_deg;
        int16_t distance = 1000;
        int16_t center_distance = 1000;
        int16_t height = 0;
    };

    struct VisionConfig {
        int frame_width = 640;
        int frame_height = 480;
        Point center{320, 240};
        bool dist_to_center = false;
    };

    extern VisionConfig vision_cfg;

    auto FindBlobs(
        const cv::Mat& lab, const std::vector<ColorThreshold>&
    ) -> std::vector<std::vector<BlobGeom>>;

    template <typename Comp>
    auto FindBlobs(
        const cv::Mat& lab, const std::vector<ColorThreshold>& thr,
        Comp&& comp,
        size_t max_blobs_per_color = 5
    ) -> std::vector<std::vector<BlobGeom>> {
        std::vector<std::vector<BlobGeom>> blobs = FindBlobs(lab, thr);
        for (auto& vec : blobs) {
            std::sort(vec.begin(), vec.end(), comp);
            vec.resize(std::min(max_blobs_per_color, vec.size()));
        }
        return blobs;
    }

    auto FindBoundingRects(const cv::Mat& mask)
        -> std::vector<BlobGeom>;

    auto FindBoundingPolygons(const cv::Mat& mask, int k = 4)
        -> std::vector<std::vector<cv::Point>>;

    BlobInfo CalcBlobInfo(const BlobGeom&);

    void CalcAngleRange(const BlobGeom& blob, BlobInfo&);
    void CalcBlobDistance(const BlobGeom& blob, BlobInfo&, bool to_center);
    void CalcBlobHeight(const BlobGeom& blob, BlobInfo&);


    class FrameFetcher {
        #ifdef DESKTOP_DEBUG
        cv::VideoCapture cap;
        cv::Mat result;
        #else

        int width = 1920;
        int height = 1080;

        char fps_text[16];
        float fps = 0;

        RK_S32 s32Ret = 0;

        VENC_STREAM_S stFrame;
        RK_U64 H264_PTS = 0;
        RK_U32 H264_TimeRef = 0; 
        VIDEO_FRAME_INFO_S stViFrame;

        MB_POOL_CONFIG_S PoolCfg;
        MB_POOL src_Pool;

        MB_BLK src_Blk;

        unsigned char* data;

        VIDEO_FRAME_INFO_S h264_frame;

        rtsp_demo_handle g_rtsplive = NULL;
        rtsp_session_handle g_rtsp_session;

        #endif

        cv::Mat frame;
        cv::Mat lab;
        cv::Mat mask;

        static constexpr int max_color_blobs_ = 2;

    public:
        FrameFetcher();

        auto ReadBlobs(const std::vector<ColorThreshold>&)
            -> std::vector<std::vector<BlobGeom>>;

        #ifdef DESKTOP_DEBUG
        void DrawRay(cv::Mat& result, const Segment& segm,
                     cv::Scalar color = cv::Scalar(0, 0, 255), int width = 1) {
            std::cout << PointToImage(segm.Begin()) << "\n";
            std::cout << PointToImage(segm.End()) << "\n";
            cv::line(
                result,
                PointToImage(segm.Begin()),
                PointToImage(segm.End()),
                color, width
            );
        }
        void DrawBlob(cv::Mat&, const BlobGeom&);
        #endif

        ~FrameFetcher();
    };

} // namespace ww_vision

#endif
