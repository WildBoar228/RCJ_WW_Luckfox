add_library(
    rcj_ww_vision_lib
    rcj_vision.cpp
)

target_include_directories(
    rcj_ww_vision_lib
    PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rcj_ww_vision_lib ${OpenCV_LIBS})
