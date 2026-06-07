#include <iostream>
#include <vector>

#include "rcj_vision.hpp"
#include "blob_sender.hpp"

namespace ww_vision {

    void SendBlobs(std::ostream& out, const std::vector<ww_vision::Blob>& blobs) {
        out << blobs.size() << " blobs\n";
        for (const ww_vision::Blob& b : blobs) {
            BlobInfo bi = CalcBlobInfo(b);

            // #ifdef DESKTOP_DEBUG
            out << "\n Blob:\n";
            out << " > left_angle: " << bi.left_angle << '\n';
            out << " > right_angle: " << bi.right_angle << '\n';
            out << " > center_angle: " << bi.center_angle << '\n';
            out << " > width: " << bi.width << '\n';
            out << " > clos_angle: " << bi.clos_angle << '\n';
            out << " > distance: " << bi.distance << '\n';
            out << " > center_distance: " << bi.center_distance << '\n';
            out << " > height: " << bi.height << '\n';
            out << std::endl;
            // #endif
        }
    }

} // namespace ww_vision
