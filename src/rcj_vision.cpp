#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

#include "geometry.hpp"
#include "rcj_vision.hpp"
#include "luckfox_mpi.h"

namespace ww_vision {

    VisionConfig vision_cfg;

    Point PointFromImage(const cv::Point2f& p) {
        return Point{
            .x = static_cast<int>(p.x),
            .y = static_cast<int>(vision_cfg.frame_height - p.y) // inverse Y axis
        };
    }

    cv::Point2f PointToImage(const Point& p) {
        return cv::Point2f(
            (p.x),
            vision_cfg.frame_height - p.y // inverse Y axis
        );
    }

    auto FindBlobs(
        const cv::Mat& lab,
        const std::vector<ColorThreshold>& thresholds
    ) -> std::vector<std::vector<BlobGeom>> {

        std::vector<std::vector<BlobGeom>> blobs;
        cv::Mat mask;

        for (const ColorThreshold& thr : thresholds) {
            cv::inRange(lab, thr.lower, thr.upper, mask);
            std::vector<BlobGeom> rects = FindBoundingRects(mask);
            blobs.push_back(std::move(rects));
        }

        return blobs;
    }

    auto FindBoundingRects(const cv::Mat& mask) -> std::vector<BlobGeom> {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(
            mask,
            contours,
            hierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        std::vector<BlobGeom> blobs;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) {
                continue;
            }
            cv::RotatedRect rect = cv::minAreaRect(contour);
            cv::Point2f corners[BlobGeom::vert_cnt];
            rect.points(corners);
            BlobGeom blob;
            for (int i = 0; i < BlobGeom::vert_cnt; ++i) {
                blob.p[i] = PointFromImage(corners[i]);
            }
            blob.center = PointFromImage(rect.center);
            blob.area = rect.size.area();
            blobs.push_back(blob);
        }

        return blobs;
    }

    BlobInfo CalcBlobInfo(const BlobGeom& blob) {
        BlobInfo bi;
        CalcAngleRange(blob, bi);
        CalcBlobDistance(blob, bi, vision_cfg.dist_to_center);
        CalcBlobHeight(blob, bi);
        return bi;
    }

    void CalcAngleRange(const BlobGeom& blob, BlobInfo& bi) {
        Deg angles[blob.vert_cnt];
        for (int i = 0; i < blob.vert_cnt; ++i) {
            angles[i] = Rad(
                std::atan2(blob.p[i].x - vision_cfg.center.x,
                           blob.p[i].y - vision_cfg.center.y)
            );
        }

        for (int i = 0; i < BlobGeom::vert_cnt; ++i) {
            if (i == 0) {
                bi.center_angle = angles[i];
                bi.left_angle = angles[i];
                bi.right_angle = angles[i];
            } else {
                if (FitAngle(angles[i] - bi.center_angle) < 0_deg) {
                    if (FitAngle(angles[i] - bi.center_angle) < 
                        FitAngle(bi.left_angle - bi.center_angle)) {
                        bi.left_angle = angles[i];
                    }
                } else {
                    if (FitAngle(angles[i] - bi.center_angle) > 
                        FitAngle(bi.right_angle - bi.center_angle)) {
                        bi.right_angle = angles[i];
                    }
                }
            }
        }
        bi.center_angle = Rad(std::atan2(
            (sin(bi.left_angle) + sin(bi.right_angle)) / 2,
            (cos(bi.left_angle) + cos(bi.right_angle)) / 2
        ));
        bi.width = bi.right_angle - bi.left_angle;
        if (bi.width < 0_deg) {
            bi.width = bi.width + 360_deg;
        }
    }

    void CalcBlobDistance(const BlobGeom& blob, BlobInfo& bi, bool to_center) {
        if (to_center) {
            bi.distance = CalcPolygonDist(
                Segment(
                    vision_cfg.center,
                    bi.center_angle,
                    vision_cfg.frame_width
                ),
                blob
            );
        } else {
            bi.distance = 1000;
            for (int i = 0; i < blob.vert_cnt; ++i) {
                int temp = CalcSegmentDist(
                    vision_cfg.center,
                    Segment(blob.p[i], blob.p[(i + 1) % blob.vert_cnt])
                );
                if (temp < bi.distance) {
                    bi.distance = temp;
                }
            }
        }
        
        bi.clos_angle = bi.center_angle;
        bi.center_distance = CalcPointDistance(vision_cfg.center, blob.center);
    }

    void CalcBlobHeight(const BlobGeom& blob, BlobInfo& bi) {
        bi.height = 0;
        for (int i = 0; i < blob.vert_cnt; ++i) {
            int temp = CalcPointDistance(vision_cfg.center, blob.p[i]);
            if (temp > bi.height) {
                bi.height = temp;
            }
        }
    }

    auto FindBoundingPolygons(const cv::Mat& mask, int k)
        -> std::vector<std::vector<cv::Point>> {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(
            mask,
            contours,
            hierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        std::vector<std::vector<cv::Point>> polygons;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) {
                continue;
            }
            
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);

            std::vector<cv::Point> poly;

            if (k <= 0) {
                double eps = 0.02 * cv::arcLength(hull, true);
                cv::approxPolyDP(hull, poly, eps, true);
            } else { 
                double perimeter = cv::arcLength(hull, true);

                for (double factor = 0.001; factor <= 0.2; factor += 0.001) {
                    std::vector<cv::Point> candidate;

                    cv::approxPolyDP(
                        hull,
                        candidate,
                        factor * perimeter,
                        true
                    );

                    if ((int)candidate.size() <= k) {
                        poly = std::move(candidate);
                        break;
                    }
                }

                if (poly.empty()) {
                    poly = std::move(hull);
                }
            }

            polygons.push_back(std::move(poly));
        }

        return polygons;
    }

    #ifdef DESKTOP_DEBUG
    FrameFetcher::FrameFetcher(): cap(0) { }

    auto FrameFetcher::ReadBlobs(const std::vector<ColorThreshold>& thresholds)
        -> std::vector<std::vector<BlobGeom>> {
        
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Empty frame\n";
            return {};
        }

        result = frame.clone();

        cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
        if (thresholds.size() >= 1) {
            cv::inRange(lab, thresholds[0].lower, thresholds[0].upper, mask);
        }

        auto blobs = FindBlobs(
            lab, thresholds,
            [](const BlobGeom& a, const BlobGeom& b) {
                return a.area > b.area;
            }, max_color_blobs_
        );

        for (const auto& color_blobs : blobs) {
            for (const BlobGeom& blob : color_blobs) {
                DrawBlob(result, blob);
            }
        }

        cv::imshow("Camera", frame);
        cv::imshow("Mask", mask);
        cv::imshow("Detected", result);
        
        std::getchar();

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            std::exit(0);
        }

        return blobs;
    }
    
    void FrameFetcher::DrawBlob(cv::Mat& result, const BlobGeom& blob) {
        std::vector<cv::Point> polygon(blob.vert_cnt);
        for (int i = 0; i < blob.vert_cnt; ++i) {
            polygon[i] = PointToImage(blob.p[i]);
        }
        cv::polylines(result, polygon, true, cv::Scalar(0, 255, 0), 2);

        BlobInfo bi = CalcBlobInfo(blob);

        DrawRay(
            result,
            Segment(
                vision_cfg.center,
                bi.center_angle,
                bi.center_distance
            ),
            cv::Scalar(100, 100, 100), 3
        );

        DrawRay(
            result,
            Segment(
                vision_cfg.center,
                bi.left_angle,
                bi.center_distance
            ),
            cv::Scalar(0, 0, 255), 2
        );

        DrawRay(
            result,
            Segment(
                vision_cfg.center,
                bi.right_angle,
                bi.center_distance
            ),
            cv::Scalar(0, 0, 255), 1
        );
    }
    
    FrameFetcher::~FrameFetcher() { }

#else

    FrameFetcher::FrameFetcher() {
        system("RkLunch-stop.sh");

        memset(fps_text,0,16);

        //h264_frame
        stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));

        // Create Pool
        memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
        PoolCfg.u64MBSize = width * height * 3 ;
        PoolCfg.u32MBCnt = 1;
        PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
        //PoolCfg.bPreAlloc = RK_FALSE;
        src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
        printf("Create Pool success !\n");	

        // Get MB from Pool 
        src_Blk = RK_MPI_MB_GetMB(src_Pool, width * height * 3, RK_TRUE);

        // Build h264_frame
        h264_frame.stVFrame.u32Width = width;
        h264_frame.stVFrame.u32Height = height;
        h264_frame.stVFrame.u32VirWidth = width;
        h264_frame.stVFrame.u32VirHeight = height;
        h264_frame.stVFrame.enPixelFormat =  RK_FMT_RGB888; 
        h264_frame.stVFrame.u32FrameFlag = 160;
        h264_frame.stVFrame.pMbBlk = src_Blk;
        data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
        frame = cv::Mat(cv::Size(width,height),CV_8UC3,data);

        // rkaiq init
        RK_BOOL multi_sensor = RK_FALSE;	
        const char *iq_dir = "/etc/iqfiles";
        rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
        //hdr_mode = RK_AIQ_WORKING_MODE_ISP_HDR2;
        SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
        SAMPLE_COMM_ISP_Run(0);

        // rkmpi init
        if (RK_MPI_SYS_Init() != RK_SUCCESS) {
            RK_LOGE("rk mpi sys init fail!");
            std::exit(-1);
        }

        // rtsp init	
        g_rtsplive = create_rtsp_demo(554);
        g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
        rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
        rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

        // vi init
        vi_dev_init();
        vi_chn_init(0, width, height);

        // venc init
        RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
        venc_init(0, width, height, enCodecType);
        
        printf("init success\n");	
    }

    auto FrameFetcher::ReadBlobs(const std::vector<ColorThreshold>& thresholds)
        -> std::vector<std::vector<BlobGeom>> {
        
        // get vi frame
		h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
		h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs(); 
		s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);

			cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
			cv::Mat bgr(height, width, CV_8UC3, data);			
			cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);
			cv::resize(bgr, frame, cv::Size(width ,height), 0, 0, cv::INTER_LINEAR);
			
			sprintf(fps_text,"fps = %.2f",fps);		
            cv::putText(frame,fps_text,
							cv::Point(40, 40),
							cv::FONT_HERSHEY_SIMPLEX,1,
							cv::Scalar(0,255,0),2);
			
		}
		memcpy(data, frame.data, width * height * 3);
		
		// encode H264	
		RK_MPI_VENC_SendFrame(0,  &h264_frame ,-1);
	
		// rtsp
		s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);	
		if(s32Ret == RK_SUCCESS) {
			if(g_rtsplive && g_rtsp_session) {
				//printf("len = %d PTS = %d \n",stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);	
				void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
				rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len,
							  stFrame.pstPack->u64PTS);
				rtsp_do_event(g_rtsplive);
			}
			RK_U64 nowUs = TEST_COMM_GetNowUs();
			fps = (float) 1000000 / (float)(nowUs - h264_frame.stVFrame.u64PTS);			
		}

		// release frame 
		s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
		}
		s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
		}

        cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
        if (thresholds.size() >= 1) {
            cv::inRange(lab, thresholds[0].lower, thresholds[0].upper, mask);
        }

        auto blobs = FindBlobs(
            lab, thresholds,
            [](const BlobGeom& a, const BlobGeom& b) {
                return a.area > b.area;
            }, max_color_blobs_
        );

        return blobs;
    }

    FrameFetcher::~FrameFetcher() {
        // Destory MB
        RK_MPI_MB_ReleaseMB(src_Blk);
        // Destory Pool
        RK_MPI_MB_DestroyPool(src_Pool);

        RK_MPI_VI_DisableChn(0, 0);
        RK_MPI_VI_DisableDev(0);
            
        SAMPLE_COMM_ISP_Stop(0);

        RK_MPI_VENC_StopRecvFrame(0);
        RK_MPI_VENC_DestroyChn(0);

        free(stFrame.pstPack);

        if (g_rtsplive)
            rtsp_del_demo(g_rtsplive);
        
        RK_MPI_SYS_Exit();
    }

    #endif

} // namespace ww_vision
