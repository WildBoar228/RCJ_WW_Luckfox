#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "rcj_vision.hpp"
#include "blob_sender.hpp"

template <typename... Args>
void DrawSegment(cv::Mat& result, const ww_vision::Segment& segm, Args&&... args) {
    std::cout << ww_vision::PointToImage(segm.Begin()) << "\n";
    std::cout << ww_vision::PointToImage(segm.End()) << "\n";
    cv::line(
        result,
        ww_vision::PointToImage(segm.Begin()),
        ww_vision::PointToImage(segm.End()),
        std::forward<Args>(args)...
    );
}

void DrawBlob(cv::Mat& result, const ww_vision::Blob& blob) {
    std::vector<cv::Point> polygon(blob.vert_cnt);
    for (int i = 0; i < blob.vert_cnt; ++i) {
        polygon[i] = ww_vision::PointToImage(blob.p[i]);
    }
    cv::polylines(result, polygon, true, cv::Scalar(0, 255, 0), 2);

    ww_vision::BlobInfo bi = ww_vision::CalcBlobInfo(blob);

    DrawSegment(
        result,
        ww_vision::Segment(
            ww_vision::vision_cfg.center,
            bi.center_angle,
            bi.center_distance
        ),
        cv::Scalar(100, 100, 100), 3
    );

    DrawSegment(
        result,
        ww_vision::Segment(
            ww_vision::vision_cfg.center,
            bi.left_angle,
            bi.center_distance
        ),
        cv::Scalar(0, 0, 255), 2
    );

    DrawSegment(
        result,
        ww_vision::Segment(
            ww_vision::vision_cfg.center,
            bi.right_angle,
            bi.center_distance
        ),
        cv::Scalar(0, 0, 255), 1
    );
}


int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera\n";
        return 1;
    }

    cv::Mat frame;
    cv::Mat lab;
    cv::Mat mask;
    cv::Mat result;
    
    std::vector<ww_vision::ColorThreshold> thresholds = {
        {
            ww_vision::ColorLab(100, 140, 140),
            ww_vision::ColorLab(150, 200, 200)
        }
    };

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Empty frame\n";
            break;
        }

        std::cout << frame.size() << "\n";
        result = frame.clone();

        cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
        std::vector<ww_vision::Blob> blobs = ww_vision::FindBlobs(
            lab, thresholds,
            [](ww_vision::Blob& a, ww_vision::Blob& b) {
                return a.area > b.area;
            }, 1
        )[0];

        for (ww_vision::Blob& blob : blobs) {
            DrawBlob(result, blob);
        }

        ww_vision::SendBlobs(std::cout, blobs);

        cv::inRange(lab, thresholds[0].lower, thresholds[0].upper, mask);

        // result.setTo(cv::Scalar(0, 0, 255), mask);

        cv::imshow("Camera", frame);
        cv::imshow("Mask", mask);
        cv::imshow("Detected", result);

        std::getchar();

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            break;
        }
    }

    return 0;
}
