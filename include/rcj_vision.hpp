#ifndef _RCJ_WW_LUCKFOX_RCJ_VISION_HPP_
#define _RCJ_WW_LUCKFOX_RCJ_VISION_HPP_

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifdef DESKTOP_DEBUG
#include <opencv2/videoio.hpp>
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
        cv::Mat mask;
        #endif

        cv::Mat frame;
        cv::Mat lab;

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
    };

} // namespace ww_vision

#endif
