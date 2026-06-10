#include <iostream>
#include <fstream>
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
    #else
    std::ofstream uart("/dev/ttyS3", std::ios::out | std::ios::binary);
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
        #ifdef DESKTOP_DEBUG
        ww_vision::SendBlobs(std::cout, blobs);
        #else
        ww_vision::SendBlobs(uart, blobs);
        #endif
    }

    return 0;
}
