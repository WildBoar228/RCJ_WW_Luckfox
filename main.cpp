#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    std::cout << cv::getBuildInformation() << '\n';
    cv::Mat frame;
    frame = cv::imread("images.jpeg", 1);
    
    #ifdef DESKTOP_DEBUG
    cv::namedWindow("debug");
    cv::imshow("debug", frame);
    #endif
    cv::waitKey(0);

    return 0;
}
