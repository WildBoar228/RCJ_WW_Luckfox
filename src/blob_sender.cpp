#include <fstream>
#include <iostream>
#include <vector>

#include "rcj_vision.hpp"
#include "blob_sender.hpp"

namespace ww {
namespace vision {

    uint16_t FitAngle(uint16_t angle) {
        angle = (angle % 360 + 360) % 360;
        if (angle > 180) {
            angle -= 360;
        }
        return angle;
    }

    // Big-endian, 2 байта, старший бит под знак
    void SerializeInt(char* data, int val) {
        data[0] = (abs(val) >> 8) & 0xFF;
        data[1] = abs(val) & 0xFF;
        if (val < 0) {
            data[0] |= 0x80;
        }
    }

    void SerializeBlob(char* data, const BlobInfo& bi) {
        SerializeInt(&data[0], bi.left_angle.deg);
        SerializeInt(&data[2], bi.center_angle.deg);
        SerializeInt(&data[4], bi.right_angle.deg);
        SerializeInt(&data[6], bi.clos_angle.deg);
        SerializeInt(&data[8], bi.distance);
        SerializeInt(&data[10], bi.width.deg);
        SerializeInt(&data[12], bi.height);
    }

    void ApplyOffsetToAngle(Deg& angle) {
        if (angle == 360_deg) {
            return;
        }

        if (vision_cfg.invert_gate_angle) {
            angle = FitAngle(-angle);
        }

        angle = FitAngle(angle + vision_cfg.gate_angle_offset);
    }

    void ApplyOffsetToBlob(BlobInfo& bi) {
        ApplyOffsetToAngle(bi.left_angle);
        ApplyOffsetToAngle(bi.right_angle);
        ApplyOffsetToAngle(bi.center_angle);
        ApplyOffsetToAngle(bi.clos_angle);

        if (vision_cfg.invert_gate_angle) {
            std::swap(bi.left_angle, bi.right_angle);
        }
    }

    // для того чтоб никто случайно не убил отправку сделаю отдельную функцию
    // так же оно надо для UART-test-mode
    void Send2BlobInfo(std::ostream& out,  BlobInfo b1,  BlobInfo b2){
        static constexpr int kSendColors = 2;
        static constexpr int kBlobInfoLen = sizeof(int16_t) * 7;
        static constexpr int kPackageLen = 2 + kSendColors * kBlobInfoLen;
        static char data[kPackageLen];
        memset(data, 0, sizeof(data));
        data[0] = data[1] = 0xFF;
        int write_index = 2;

        ApplyOffsetToBlob(b1);
        ApplyOffsetToBlob(b2);

        SerializeBlob(&data[write_index], b1);
        write_index += kBlobInfoLen;
        SerializeBlob(&data[write_index], b2);

        out.write(data, sizeof(data));
        out.flush();
    }

    void SendBlobs(
        std::ostream& out,
        const std::vector<std::vector<BlobGeom>>& blobs) {
        static constexpr int kSendColors = 2;
        BlobInfo BlobInfoMass[kSendColors];

        int colors_to_process = std::min(kSendColors, static_cast<int>(blobs.size()));

        for (int i = 0; i < colors_to_process; ++i) {
            const auto& color_blobs = blobs[i];
            if (!color_blobs.empty()) {
                BlobInfo bi = CalcBlobInfo(color_blobs[0]);
                BlobInfoMass[i] = bi;
            }
        }
        

        for (const auto& color_blobs : blobs) {
            if (!color_blobs.empty()) {
                std::cout << color_blobs.size() << " blobs\n";
            }
            for (const BlobGeom& b : color_blobs) {
                BlobInfo bi = CalcBlobInfo(b);
                std::cout << "\n BlobGeom:\n";
                std::cout << " > left_angle: " << bi.left_angle << '\n';
                std::cout << " > right_angle: " << bi.right_angle << '\n';
                std::cout << " > center_angle: " << bi.center_angle << '\n';
                std::cout << " > width: " << bi.width << '\n';
                std::cout << " > clos_angle: " << bi.clos_angle << '\n';
                std::cout << " > distance: " << bi.distance << '\n';
                std::cout << " > center_distance: " << bi.center_distance << '\n';
                std::cout << " > height: " << bi.height << '\n';
                std::cout << std::endl;
            }
        }

        #ifdef DESKTOP_DEBUG

        #else

        Send2BlobInfo(out, BlobInfoMass[0], BlobInfoMass[1]);
        
        #endif
    }

} // namespace vision
} // namespace ww
