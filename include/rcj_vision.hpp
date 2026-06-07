#ifndef _RCJ_WW_LUCKFOX_RCJ_VISION_HPP_
#define _RCJ_WW_LUCKFOX_RCJ_VISION_HPP_

#include <cstdint>
#include <opencv2/core.hpp>

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
        const cv::Mat& lab, std::vector<ColorThreshold>&
    ) -> std::vector<std::vector<Blob>>;

    template <typename Comp>
    auto FindBlobs(
        const cv::Mat& lab, std::vector<ColorThreshold>& thr,
        Comp&& comp,
        size_t max_blobs_per_color = 5
    ) -> std::vector<std::vector<Blob>> {
        std::vector<std::vector<Blob>> blobs = FindBlobs(lab, thr);
        for (auto& vec : blobs) {
            std::sort(vec.begin(), vec.end(), comp);
            vec.resize(std::min(max_blobs_per_color, vec.size()));
        }
        return blobs;
    }

    auto FindBoundingRects(const cv::Mat& mask)
        -> std::vector<Blob>;

    auto FindBoundingPolygons(const cv::Mat& mask, int k = 4)
        -> std::vector<std::vector<cv::Point>>;

    BlobInfo CalcBlobInfo(const Blob&);

    void CalcAngleRange(const Blob& blob, BlobInfo&);
    void CalcBlobDistance(const Blob& blob, BlobInfo&, bool to_center);
    void CalcBlobHeight(const Blob& blob, BlobInfo&);

} // namespace ww_vision

#endif
