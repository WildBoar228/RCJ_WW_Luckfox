#include <iostream>
#include <utility>
#include <vector>

#ifdef DESKTOP_DEBUG
#include <opencv2/videoio.hpp>
#endif

#include "rcj_vision.hpp"
#include "blob_sender.hpp"


int main() {
    #ifdef DESKTOP_DEBUG
    std::cout << "BUILD_DESKTOP_DEBUG\n";
    #endif
    ww_vision::FrameFetcher ff;
    
    std::vector<ww_vision::ColorThreshold> thresholds = {
        {
            ww_vision::ColorLab(100, 140, 140),
            ww_vision::ColorLab(150, 200, 200)
        }
    };

    while (true) {
        auto blobs = ff.ReadBlobs(thresholds);
        for (const auto& color_blobs : blobs) {
            ww_vision::SendBlobs(std::cout, color_blobs);
        }
        std::getchar();

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            break;
        }
    }

    return 0;
}
