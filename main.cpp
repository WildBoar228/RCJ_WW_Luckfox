#include <iostream>

#include <opencv2/opencv.hpp>

#include "geometry.hpp"

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Capture failed\n";
            break;
        }

        std::cout << frame.cols << "x" << frame.rows << std::endl;

        #ifdef DESKTOP_DEBUG
        cv::imshow("camera", frame);
        if (cv::waitKey(1) == 27) {
            break;
        }
        #endif
    }

    return 0;
}
