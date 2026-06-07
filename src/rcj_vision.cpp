#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

#include "geometry.hpp"
#include "rcj_vision.hpp"

namespace ww_vision {

    VisionConfig vision_cfg;

    Point PointFromImage(const cv::Point2f& p) {
        return Point{
            .x = static_cast<int>(p.x),
            .y = static_cast<int>(vision_cfg.frame_height - p.y) // inverse Y axis
        };
    }

    cv::Point2f PointToImage(const Point& p) {
        return cv::Point2f(
            (p.x),
            vision_cfg.frame_height - p.y // inverse Y axis
        );
    }

    auto FindBlobs(
        const cv::Mat& lab,
        std::vector<ColorThreshold>& thresholds
    ) -> std::vector<std::vector<Blob>> {

        std::vector<std::vector<Blob>> blobs;
        cv::Mat mask;

        for (ColorThreshold& thr : thresholds) {
            cv::inRange(lab, thr.lower, thr.upper, mask);
            std::vector<Blob> rects = FindBoundingRects(mask);
            blobs.push_back(std::move(rects));
        }

        return blobs;
    }

    auto FindBoundingRects(const cv::Mat& mask) -> std::vector<Blob> {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(
            mask,
            contours,
            hierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        std::vector<Blob> blobs;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) {
                continue;
            }
            cv::RotatedRect rect = cv::minAreaRect(contour);
            cv::Point2f corners[Blob::vert_cnt];
            rect.points(corners);
            Blob blob;
            for (int i = 0; i < Blob::vert_cnt; ++i) {
                blob.p[i] = PointFromImage(corners[i]);
            }
            blob.center = PointFromImage(rect.center);
            blob.area = rect.size.area();
            blobs.push_back(blob);
        }

        return blobs;
    }

    BlobInfo CalcBlobInfo(const Blob& blob) {
        BlobInfo bi;
        CalcAngleRange(blob, bi);
        CalcBlobDistance(blob, bi, vision_cfg.dist_to_center);
        CalcBlobHeight(blob, bi);
        return bi;
    }

    void CalcAngleRange(const Blob& blob, BlobInfo& bi) {
        Deg angles[blob.vert_cnt];
        for (int i = 0; i < blob.vert_cnt; ++i) {
            angles[i] = Rad(
                std::atan2(blob.p[i].x - vision_cfg.center.x,
                           blob.p[i].y - vision_cfg.center.y)
            );
            std::cout << i << ":  corner angle " << angles[i] << "\n";
        }

        for (int i = 0; i < Blob::vert_cnt; ++i) {
            if (i == 0) {
                bi.center_angle = angles[i];
                bi.left_angle = angles[i];
                bi.right_angle = angles[i];
            } else {
                if (FitAngle(angles[i] - bi.center_angle) < 0_deg) {
                    if (FitAngle(angles[i] - bi.center_angle) < 
                        FitAngle(bi.left_angle - bi.center_angle)) {
                        bi.left_angle = angles[i];
                    }
                } else {
                    if (FitAngle(angles[i] - bi.center_angle) > 
                        FitAngle(bi.right_angle - bi.center_angle)) {
                        bi.right_angle = angles[i];
                    }
                }
                std::cout << angles[i] << ":   " << bi.left_angle << " < " << bi.center_angle << " < " << bi.right_angle << "\n";
            }
        }
        bi.center_angle = Rad(std::atan2(
            (sin(bi.left_angle) + sin(bi.right_angle)) / 2,
            (cos(bi.left_angle) + cos(bi.right_angle)) / 2
        ));
        bi.width = bi.right_angle - bi.left_angle;
        if (bi.width < 0_deg) {
            bi.width = bi.width + 360_deg;
        }
    }

    void CalcBlobDistance(const Blob& blob, BlobInfo& bi, bool to_center) {
        if (to_center) {
            bi.distance = CalcPolygonDist(
                Segment(
                    vision_cfg.center,
                    bi.center_angle,
                    vision_cfg.frame_width
                ),
                blob
            );
        } else {
            bi.distance = 1000;
            for (int i = 0; i < blob.vert_cnt; ++i) {
                int temp = CalcSegmentDist(
                    vision_cfg.center,
                    Segment(blob.p[i], blob.p[(i + 1) % blob.vert_cnt])
                );
                if (temp < bi.distance) {
                    bi.distance = temp;
                }
            }
        }
        
        bi.clos_angle = bi.center_angle;
        bi.center_distance = ww_vision::CalcPointDistance(vision_cfg.center, blob.center);
    }

    void CalcBlobHeight(const Blob& blob, BlobInfo& bi) {
        bi.height = 0;
        for (int i = 0; i < blob.vert_cnt; ++i) {
            int temp = CalcPointDistance(vision_cfg.center, blob.p[i]);
            if (temp > bi.height) {
                bi.height = temp;
            }
        }
    }

    auto FindBoundingPolygons(const cv::Mat& mask, int k)
        -> std::vector<std::vector<cv::Point>> {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(
            mask,
            contours,
            hierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        std::vector<std::vector<cv::Point>> polygons;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) {
                continue;
            }
            
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);

            std::vector<cv::Point> poly;

            if (k <= 0) {
                double eps = 0.02 * cv::arcLength(hull, true);
                cv::approxPolyDP(hull, poly, eps, true);
            } else { 
                double perimeter = cv::arcLength(hull, true);

                for (double factor = 0.001; factor <= 0.2; factor += 0.001) {
                    std::vector<cv::Point> candidate;

                    cv::approxPolyDP(
                        hull,
                        candidate,
                        factor * perimeter,
                        true
                    );

                    if ((int)candidate.size() <= k) {
                        poly = std::move(candidate);
                        break;
                    }
                }

                if (poly.empty()) {
                    poly = std::move(hull);
                }
            }

            polygons.push_back(std::move(poly));
        }

        return polygons;
    }

} // namespace ww_vision
