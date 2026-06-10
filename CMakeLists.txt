cmake_minimum_required(VERSION 3.20)

project(
    rcj_ww_vision
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OPENCV_MOBILE_PATH $ENV{OPENCV_MOBILE_PATH})
set(LUCKFOX_C_CROSS_COMPILER_PATH $ENV{LUCKFOX_C_CROSS_COMPILER_PATH})
set(LUCKFOX_CXX_CROSS_COMPILER_PATH $ENV{LUCKFOX_CXX_CROSS_COMPILER_PATH})
set(LUCKFOX_RKMPI_PATH $ENV{LUCKFOX_RKMPI_PATH})

option(BUILD_DESKTOP_DEBUG "Build for desktop" OFF)
option(BUILD_VISION_TESTS "Enable testing" OFF)

if(BUILD_DESKTOP_DEBUG)
    find_package(OpenCV REQUIRED)
else()
    if(NOT DEFINED LUCKFOX_C_CROSS_COMPILER_PATH
       OR "${LUCKFOX_CXX_CROSS_COMPILER_PATH}" STREQUAL "")
        message(FATAL_ERROR
            "LUCKFOX_C_CROSS_COMPILER_PATH environment variable is not set.\n"
            "Update your .env (see .env.example) and export it")
    endif()

    if(NOT DEFINED LUCKFOX_CXX_CROSS_COMPILER_PATH
       OR "${LUCKFOX_CXX_CROSS_COMPILER_PATH}" STREQUAL "")
        message(FATAL_ERROR
            "LUCKFOX_CXX_CROSS_COMPILER_PATH environment variable is not set.\n"
            "Update your .env (see .env.example) and export it")
    endif()

    if(NOT DEFINED OPENCV_MOBILE_PATH
       OR "${OPENCV_MOBILE_PATH}" STREQUAL "")
        message(FATAL_ERROR
            "OPENCV_MOBILE_PATH environment variable is not set.\n"
            "Update your .env (see .env.example) and export it")
    endif()
    
    set(CMAKE_C_COMPILER "${LUCKFOX_C_CROSS_COMPILER_PATH}")
    set(CMAKE_CXX_COMPILER "${LUCKFOX_CXX_CROSS_COMPILER_PATH}")

    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/lib:/usr/lib")

    add_definitions(-DRV1106_1103)
    link_directories(${LUCKFOX_RKMPI_PATH}/lib/uclibc)

    set(OpenCV_DIR "${LUCKFOX_RKMPI_PATH}/lib/uclibc/lib/cmake/opencv4")
    message("${OpenCV_DIR} <- OpenCV_DIR")
    find_package(OpenCV REQUIRED
        NO_DEFAULT_PATH
        PATHS ${OpenCV_DIR})

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)

    add_compile_options(-g -Wall
                        -DISP_HW_V30 -DRKPLATFORM=ON -DARCH64=OFF
                        # -DROCKIVA
                        -DUAPI2
                        -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 
                        )
endif()

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

add_subdirectory(src)

if(BUILD_DESKTOP_DEBUG)
    message("BUILD_DESKTOP_DEBUG")
    target_compile_definitions(rcj_ww_vision_lib PRIVATE DESKTOP_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DESKTOP_DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE "-fsanitize=address,leak,undefined")
    target_link_options(${PROJECT_NAME} PRIVATE "-fsanitize=address,leak,undefined")

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include)

    target_link_libraries(
        ${PROJECT_NAME}
        rcj_ww_vision_lib
        ${OpenCV_LIBS}
    )
else()

    target_link_libraries(${PROJECT_NAME} rcj_ww_vision_lib ${OpenCV_LIBS})

    target_include_directories(${PROJECT_NAME} PRIVATE                        
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

if (BUILD_VISION_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE -O2)
