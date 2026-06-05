include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    FIND_PACKAGE_ARGS NAMES GTest
)

enable_testing()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
    rcj_vision_tests
    angles_ut.cpp
    geometry_ut.cpp
)

target_link_libraries(
    rcj_vision_tests
    GTest::gtest_main
    GTest::gmock_main
    rcj_ww_vision_lib
)

target_compile_options(rcj_vision_tests PRIVATE "-fsanitize=address,leak,undefined" "-O2")
target_link_options(rcj_vision_tests PRIVATE "-fsanitize=address,leak,undefined")

target_include_directories(rcj_vision_tests PUBLIC ${PROJECT_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(rcj_vision_tests)
