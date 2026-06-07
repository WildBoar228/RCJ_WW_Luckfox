cmake_minimum_required(VERSION 3.20)

project(
    rcj_ww_vision
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_DESKTOP_DEBUG "Build for desktop" OFF)

if(BUILD_DESKTOP_DEBUG)
    find_package(OpenCV REQUIRED)
else()
    # TODO: set cross-compiler
    # set(CMAKE_C_COMPILER ..)
    # set(CMAKE_CXX_COMPILER ..)

    if(NOT DEFINED ENV{OPENCV_MOBILE_PATH}
       OR "$ENV{OPENCV_MOBILE_PATH}" STREQUAL "")
        message(FATAL_ERROR
            "OPENCV_MOBILE_PATH environment variable is not set.\n"
            "Update your .env (see .env.example) and export it")
    endif()

    set(OpenCV_DIR $ENV{OPENCV_MOBILE_PATH}/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED
        NO_DEFAULT_PATH
        PATHS ${OpenCV_DIR})
endif()

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

add_subdirectory(src)

if(BUILD_DESKTOP_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DESKTOP_DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE "-fsanitize=address,leak,undefined")
    target_link_options(${PROJECT_NAME} PRIVATE "-fsanitize=address,leak,undefined")

    enable_testing()
    add_subdirectory(tests)
endif()

target_link_libraries(
    ${PROJECT_NAME}
    rcj_ww_vision_lib
    ${OpenCV_LIBS}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE -O2)
