#include <fstream>
#include <iostream>
#include <vector>

#include "rcj_vision.hpp"
#include "blob_sender.hpp"

namespace ww_vision {

    void SerializeInt(char* data, int16_t val) {
        data[0] = (static_cast<uint16_t>(val) >> 8) & 0xFF;
        data[1] = val & 0xFF;
    }

    void SerializeBlob(char* data, const BlobInfo& bi) {
        SerializeInt(&data[0], bi.left_angle);
        SerializeInt(&data[2], bi.center_angle);
        SerializeInt(&data[4], bi.right_angle);
        SerializeInt(&data[6], bi.clos_angle);
        SerializeInt(&data[8], bi.distance);
        SerializeInt(&data[10], bi.width);
        SerializeInt(&data[12], bi.height);
    }

    void SendBlobs(
        std::ostream& out,
        const std::vector<std::vector<BlobGeom>>& blobs
    ) {
        #ifdef DESKTOP_DEBUG

        for (const auto& color_blobs : blobs) {
            out << blobs.size() << " blobs\n";
            for (const ww_vision::BlobGeom& b : color_blobs) {
                BlobInfo bi = CalcBlobInfo(b);
                out << "\n BlobGeom:\n";
                out << " > left_angle: " << bi.left_angle << '\n';
                out << " > right_angle: " << bi.right_angle << '\n';
                out << " > center_angle: " << bi.center_angle << '\n';
                out << " > width: " << bi.width << '\n';
                out << " > clos_angle: " << bi.clos_angle << '\n';
                out << " > distance: " << bi.distance << '\n';
                out << " > center_distance: " << bi.center_distance << '\n';
                out << " > height: " << bi.height << '\n';
                out << std::endl;
            }
        }

        #else

        static constexpr int kBlobInfoLen = sizeof(int16_t) * 7;
        static constexpr int kPackageLen = 2 + 2 * kBlobInfoLen;
        static char data[kPackageLen];
        memset(data, 0, sizeof(data));
        data[0] = data[1] = 0xFF;
        int write_index = 2;

        for (const auto& color_blobs : blobs) {
            if (!color_blobs.empty()) {
                BlobInfo bi = CalcBlobInfo(color_blobs[0]);
                SerializeBlob(&data[write_index], bi);
            }
            write_index += kBlobInfoLen;
        }

        out.write(data, sizeof(data));

        #endif
    }

} // namespace ww_vision
