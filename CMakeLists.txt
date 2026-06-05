cmake_minimum_required(VERSION 3.20)

project(
    rcj_ww_vision
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_DESKTOP_DEBUG "Build for Linux desktop" OFF)
set(OPENCV_MOBILE_PATH "/home/igor/Documents/cpp/libs/opencv-mobile-4.13.0-luckfox-pico")

if(BUILD_DESKTOP_DEBUG)
    find_package(OpenCV REQUIRED)
else()
    # TODO: set cross-compiler
    # set(CMAKE_C_COMPILER ..)
    # set(CMAKE_CXX_COMPILER ..)
    find_package(OpenMP REQUIRED)
    set(OpenCV_DIR ${OPENCV_MOBILE_PATH}/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
endif()

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

add_subdirectory(src)

if(BUILD_DESKTOP_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DESKTOP_DEBUG)

    enable_testing()
    add_subdirectory(tests)
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_link_libraries(
    ${PROJECT_NAME}
    rcj_ww_vision_lib
    ${OpenCV_LIBS}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE -O2)
