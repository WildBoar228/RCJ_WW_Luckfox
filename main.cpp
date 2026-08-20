#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>
#include <optional>
#include <utility>
#include <vector>

#ifdef DESKTOP_DEBUG
#include <opencv2/videoio.hpp>
#else
#include <termios.h>
#endif

#include "rcj_vision.hpp"
#include "rcj_neurovision.hpp"
#include "blob_sender.hpp"


#ifndef DESKTOP_DEBUG
int SetupUart(const char* serial_port) {
    int serial_fd = open(serial_port, O_RDWR | O_NOCTTY);
    if (serial_fd == -1) {
        perror("Failed to open serial port, NO UART!!!");
        return 1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (tcgetattr(serial_fd, &tty) != 0) {
        perror("Error from tcgetattr");
        return 1;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
        perror("Error from tcsetattr");
        return 1;
    }
    close(serial_fd);

    return 0;
}
#endif


bool GetArgvFlag(int argc, char** argv, const char* name, bool default_value = false) {
    bool value = default_value;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--no-", sizeof("--no-") - 1) == 0) {
            if (strcmp(argv[i] + sizeof("--no-") - 1, name) == 0) {
                value = false;
            }
        } else if (strncmp(argv[i], "--", sizeof("--") - 1) == 0) {
            if (strcmp(argv[i] + sizeof("--") - 1, name) == 0) {
                value = true;
            }
        }
    }
    return value;
}

const char* GetArgvString(
    int argc, char** argv, const char* name, const char* default_value) {
    const char* value = default_value;
    char* param;
    int param_name_len = strlen(name);
    for (int i = 1; i < argc; ++i) {
        param = argv[i];
        if (param[0] == '-' && param[1] == '-') {
            param += 2;
            if (strncmp(param, name, param_name_len) == 0) {
                param += param_name_len;
                if (*param == '\0') {
                    value = default_value;
                } else {
                    value = param + 1;
                }
            }
        }
    }
    return value;
}


struct RuntimeCfg {
    bool draw_blobs;
    std::vector<ww::vision::ColorThreshold> thresholds;
} runtime_cfg;

bool CheckVarName(std::istream& in, const char* expected) {
    std::string word;
    in >> word;
    if (word != expected) {
        std::cout << "Syntax error, \"" << expected << "\" expected" << std::endl;
        return false;
    }
    return true;
}

std::optional<RuntimeCfg> ReadRuntimeCfg(const char* src_name) {
    std::ifstream fin(src_name);
    if (!fin) {
        std::cout << "ERROR: can't read " << src_name << std::endl;
        return std::nullopt;
    }

    RuntimeCfg cfg;

    if (!CheckVarName(fin, "draw_blobs")) { return std::nullopt; }
    fin >> cfg.draw_blobs;

    if (!CheckVarName(fin, "thr_cnt")) { return std::nullopt; }
    int thr_cnt;
    fin >> thr_cnt;

    cfg.thresholds.resize(thr_cnt);
    for (int i = 0; i < thr_cnt; ++i) {
        if (!CheckVarName(fin, "thr")) { return std::nullopt; }
        fin >> cfg.thresholds[i];
        if (fin.fail()) { return std::nullopt; }
    }
    
    return cfg;
}


int main(int argc, char** argv) {
    using namespace ww::vision;
    vision_cfg.send_stream = GetArgvFlag(argc, argv, "stream", true);
    vision_cfg.draw_blobs = GetArgvFlag(argc, argv, "draw-blobs", true);
    bool update_runtime_cfg = GetArgvFlag(argc, argv, "runtime-cfg", true);
    const char* detect_mode = GetArgvString(argc, argv, "detect-mode", "thr-blobs");

    printf("send_stream: %d\n", vision_cfg.send_stream);
    printf("draw_blobs: %d\n", vision_cfg.draw_blobs);
    printf("detect_mode: %s\n", detect_mode);

    if (strcmp(detect_mode, "thr-blobs") == 0) {
        vision_cfg.detect_mode = DetectMode::kThresholdBlobs;
    } else if (strcmp(detect_mode, "yolo-pose") == 0) {
        #ifdef DESKTOP_DEBUG
        printf("WARNING: yolo-pose mode doesn't detect anything on DESKTOP_DEBUG\n");
        #endif
        vision_cfg.detect_mode = DetectMode::kYoloSegments;
    } else {
        printf("WARNING: Unknown detect mode \"%s\", available: thr-blobs, yolo-pose\n",
            detect_mode);
        printf("Set thr-blobs as default\n\n");
        vision_cfg.detect_mode = DetectMode::kThresholdBlobs;
    }

    #ifdef DESKTOP_DEBUG
    std::cout << "BUILD_DESKTOP_DEBUG\n";
    const char* runtime_cfg_path = "runtime.cfg";
    #else

    const char* serial_port = "/dev/ttyS3";

    int uart_err = SetupUart(serial_port);

    std::ofstream uart(serial_port, std::ios::out | std::ios::binary);
    if (uart.fail()) {
        std::cerr << "ERROR: can't write to ttyS3" << std::endl;
    }

    const char* runtime_cfg_path = "/userdata/runtime.cfg";
    const char* gate_model_path = "/root/best-int8.rknn";

    YoloConfig& yolo_cfg = yolo_config();
    yolo_cfg.class_count = ww::vision::kClassNum;
    yolo_cfg.keypoint_count = ww::vision::kKeypointNum;
    yolo_cfg.keypoint_dimensions = ww::vision::kKeypointDimensions;
    yolo_cfg.dfl_bins = ww::vision::kDflBins;
    yolo_cfg.detection_head_count = ww::vision::kDetectionHeadNum;
    yolo_cfg.keypoint_output_index = ww::vision::kDetectionHeadNum;
    yolo_cfg.anchor_count = ww::vision::kTotalAnchors;
    yolo_cfg.max_results = YoloConfig::result_capacity;
    yolo_cfg.nms_threshold = 0.4f;
    yolo_cfg.box_threshold = ww::vision::kDetectionThreshold;
    yolo_cfg.debug_output = false;
    yolo_cfg.labels_path.clear();

    GateSegmentDetector gate_detector(const_cast<char*>(gate_model_path));

    #endif

    FrameFetcher ff;
    ThresholdBlobDetector bd;
    
    std::vector<ColorThreshold> thresholds = {
        {
            ColorLab(100, 140, 140),
            ColorLab(150, 200, 200)
        }
    };

    auto clock = std::chrono::steady_clock();

    auto cfg_update_time = clock.now();
    auto cfg_update_interval = std::chrono::seconds(2);

    while (true) {
        if (update_runtime_cfg) {
            if (clock.now() - cfg_update_time > cfg_update_interval) {
                auto cfg_opt = ReadRuntimeCfg(runtime_cfg_path);
                if (!cfg_opt) {
                    std::cout << "WARNING: failed to update config\n";
                } else {
                    vision_cfg.draw_blobs = cfg_opt->draw_blobs;
                    thresholds = cfg_opt->thresholds;
                    
                    std::cout << "Thresholds updated:\n";
                    for (auto& thr : thresholds) {
                        std::cout << thr << "\n";
                    }
                    std::cout << "Draw blobs: " << vision_cfg.draw_blobs << "\n";
                }
                cfg_update_time = clock.now();
            }
        }

        ff.Fetch();

        if (vision_cfg.detect_mode == DetectMode::kThresholdBlobs) {
            auto blobs = bd.ReadBlobs(ff.GetFrame(), thresholds);
            
            #ifdef DESKTOP_DEBUG
            SendBlobs(std::cout, blobs);
            #else
            if (uart_err == 0) {
                SendBlobs(uart, blobs);
            }
            #endif

            if (vision_cfg.draw_blobs) {
                bd.DrawBlobs(ff.GetFrame(), blobs);
            }
        } else if (vision_cfg.detect_mode == DetectMode::kYoloSegments) {
            #ifndef DESKTOP_DEBUG
            auto gates_opt = gate_detector.Detect(
                ff.GetFrame(),
                ff.GetFrameFd()
            );

            if (vision_cfg.draw_blobs) {
                if (gates_opt) {
                    gate_detector.DrawResult(ff.GetFrame(), *gates_opt);
                }
            }
            #endif
        }

        if (vision_cfg.send_stream) {
            ff.SendStream();
        }
    }

    return 0;
}
