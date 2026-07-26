#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

#ifdef DESKTOP_DEBUG
#include <opencv2/videoio.hpp>
#else
#include <termios.h>
#endif

#include "rcj_vision.hpp"
#include "blob_sender.hpp"


std::vector<std::vector<int>> parsePythonList(const std::string& s) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;

    int depth = 0;
    size_t i = 0;

    while (i < s.size()) {
        char c = s[i];

        if (c == '[') {
            ++depth;

            // Начало внутреннего списка
            if (depth == 2) {
                current.clear();
            }

            ++i;
        }
        else if (c == ']') {
            // Конец внутреннего списка
            if (depth == 2) {
                result.push_back(current);
            }

            --depth;
            ++i;
        }
        else if ((c == '-') || std::isdigit(static_cast<unsigned char>(c))) {
            int sign = 1;

            if (c == '-') {
                sign = -1;
                ++i;
            }

            int value = 0;
            while (i < s.size() &&
                   std::isdigit(static_cast<unsigned char>(s[i]))) {
                value = value * 10 + (s[i] - '0');
                ++i;
            }

            // Добавляем только если находимся внутри внутреннего списка
            if (depth == 2) {
                current.push_back(sign * value);
            }
        }
        else {
            ++i; // пропускаем пробелы и запятые
        }
    }

    return result;
}

void ReadThresholds(const char* thr_path, std::vector<ww_vision::ColorThreshold>& result) {
    std::ifstream in(thr_path);
    if (in.fail()) {
        std::cerr << "ERROR reading thresholds: can't open " << thr_path << std::endl;
        return;
    }
    std::string text;
    std::string word;
    while (in >> word) {
        text += word + " ";
    }

    std::vector<std::vector<int>> colors = parsePythonList(text);

    std::vector<ww_vision::ColorThreshold> thresholds;
    for (auto& color : colors) {
        if (color.size() != 6) {
            std::cerr << "ERROR: can't read threshold: wrong count of elements ("
                      << color.size() << "): ";
            for (int c : color) {
                std::cerr << c << " ";
            }
            std::cerr << std::endl;
            return;
        }

        // L from 0..100 to 0..255
        color[0] = color[0] * 255 / 100;
        color[1] = color[1] * 255 / 100;
        // A from -128..127 to 0..255
        color[2] = color[2] + 128;
        color[3] = color[3] + 128;
        // B from -128..127 to 0..255
        color[4] = color[4] + 128;
        color[5] = color[5] + 128;

        thresholds.push_back(ww_vision::ColorThreshold{
            ww_vision::ColorLab(color[0], color[2], color[4]),
            ww_vision::ColorLab(color[1], color[3], color[5])
        });
    }

    thresholds.resize(2);
    result = thresholds;
}

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


int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--stream") == 0) {
        ww_vision::vision_cfg.send_stream = true;
        std::cout << "--stream detected\n";
    }

    #ifdef DESKTOP_DEBUG
    std::cout << "BUILD_DESKTOP_DEBUG\n";
    const char* thresholds_path = "thresholds.txt";
    #else

    const char* serial_port = "/dev/ttyS3";

    int uart_err = SetupUart(serial_port);

    std::ofstream uart(serial_port, std::ios::out | std::ios::binary);
    if (uart.fail()) {
        std::cerr << "ERROR: can't write to ttyS3" << std::endl;
    }

    const char* thresholds_path = "/userdata/thresholds.txt";
    #endif

    ww_vision::FrameFetcher ff;
    ww_vision::BlobDetector bd;
    
    std::vector<ww_vision::ColorThreshold> thresholds = {
        {
            ww_vision::ColorLab(100, 140, 140),
            ww_vision::ColorLab(150, 200, 200)
        }
    };

    auto clock = std::chrono::steady_clock();

    auto thr_update_time = clock.now();
    auto thr_update_interval = std::chrono::seconds(2);

    while (true) {
        if (clock.now() - thr_update_time > thr_update_interval) {
            ReadThresholds(thresholds_path, thresholds);
            thr_update_time = clock.now();
        }

        auto blobs = bd.ReadBlobs(ff.Fetch(), thresholds);
        #ifdef DESKTOP_DEBUG
        ww_vision::SendBlobs(std::cout, blobs);
        #else
        if (uart_err == 0) {
            ww_vision::SendBlobs(uart, blobs);
        }
        #endif
    }

    return 0;
}
