add_library(
    rcj_ww_vision_lib
    blob_sender.cpp
    rcj_neurovision.cpp
    rcj_vision.cpp
    luckfox_mpi.cc
)

if(BUILD_DESKTOP_DEBUG)
else()

    target_link_libraries(rcj_ww_vision_lib
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

    target_include_directories(rcj_ww_vision_lib PRIVATE                        
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

endif()

target_include_directories(
    rcj_ww_vision_lib
    PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rcj_ww_vision_lib ${OpenCV_LIBS})
