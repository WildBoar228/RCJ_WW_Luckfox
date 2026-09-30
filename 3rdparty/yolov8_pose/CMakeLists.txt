if(BUILD_DESKTOP_DEBUG)
else()
    add_library(yolov8_pose
        postprocess.cc
        rknpu2/yolov8-pose.cc
        image_utils.c
    )

    target_link_libraries(yolov8_pose
                        ${OpenCV_LIBS}
                        rknnmrt
                        Threads::Threads
                        rockiva
                        sample_comm
                        rockit
                        rockchip_mpp
                        rkaiq
                        pthread
                        rtsp
                        rga
                        )

    target_include_directories(yolov8_pose PRIVATE    
                                ${CMAKE_CURRENT_SOURCE_DIR}
                                ${OpenCV_INCLUDE_DIRS}
                                ${PROJECT_SOURCE_DIR}/include
                                ${LUCKFOX_RKMPI_PATH}
                                ${LUCKFOX_RKMPI_PATH}/utils
                                ${LUCKFOX_RKMPI_PATH}/common 
                                ${LUCKFOX_RKMPI_PATH}/common/isp3.x   
                                ${LUCKFOX_RKMPI_PATH}/include
                                ${LUCKFOX_RKMPI_PATH}/include/rknn
                                ${LUCKFOX_RKMPI_PATH}/include/librga 
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/uAPI2
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/common
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/xcore
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/algos
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/iq_parser
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/iq_parser_v2
                                ${LUCKFOX_RKMPI_PATH}/include/rkaiq/smartIr
                            )

    target_compile_definitions(yolov8_pose PRIVATE LIBRGA_IM2D_HANDLE)

endif()
