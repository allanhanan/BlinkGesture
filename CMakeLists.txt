cmake_minimum_required(VERSION 3.17)
project(blinkGesture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS c;C;cpp;CPP;cxx;CXX;cc;CC;h;H;hpp)

include(ExternalProject)

# === User options ===
option(USE_SYSTEM_OPENCV "Use system-installed OpenCV instead of downloading/building" OFF)
option(USE_SYSTEM_ONNX "Use system-installed ONNX Runtime instead of downloading prebuilt" OFF)
set(OPENCV_BUILD_JOBS 5 CACHE STRING "Max parallel jobs for building OpenCV")

# === Thirdparty base ===
set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/thirdparty)
file(MAKE_DIRECTORY ${THIRDPARTY_DIR})

# === ONNX Runtime ===
if(USE_SYSTEM_ONNX)
    find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_c_api.h REQUIRED)
else()
    set(ONNX_INSTALL_DIR ${THIRDPARTY_DIR}/onnx_install)
    set(ONNX_STAMP_FILE ${ONNX_INSTALL_DIR}/.downloaded)

    if(WIN32)
        set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip")
        set(ONNX_ARCHIVE "${THIRDPARTY_DIR}/onnxruntime-win.zip")
        set(ONNX_EXTRACT_DIR "${THIRDPARTY_DIR}/onnxruntime-win-x64-1.22.0")
    else()
        set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz")
        set(ONNX_ARCHIVE "${THIRDPARTY_DIR}/onnxruntime-linux.tgz")
        set(ONNX_EXTRACT_DIR "${THIRDPARTY_DIR}/onnxruntime-linux-x64-1.22.0")
    endif()

    if(NOT EXISTS "${ONNX_STAMP_FILE}")
        message(STATUS "Downloading ONNX Runtime: ${ONNX_URL}")
        file(DOWNLOAD ${ONNX_URL} ${ONNX_ARCHIVE} SHOW_PROGRESS)

        message(STATUS "Extracting ONNX Runtime...")
        if(WIN32)
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNX_ARCHIVE} WORKING_DIRECTORY ${THIRDPARTY_DIR} ERROR_QUIET RESULT_VARIABLE TAR_RESULT)
            if(NOT TAR_RESULT EQUAL 0)
                execute_process(COMMAND powershell -Command "Expand-Archive -Force '${ONNX_ARCHIVE}' '${THIRDPARTY_DIR}'")
            endif()
        else()
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNX_ARCHIVE} WORKING_DIRECTORY ${THIRDPARTY_DIR})
        endif()

        file(REMOVE_RECURSE ${ONNX_INSTALL_DIR})
        file(RENAME ${ONNX_EXTRACT_DIR} ${ONNX_INSTALL_DIR})
        file(WRITE ${ONNX_STAMP_FILE} "onnxruntime-prebuilt-downloaded")
    endif()
endif()

# === OpenCV ===
if(USE_SYSTEM_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio tracking objdetect imgcodecs)
else()
    set(OpenCV_SOURCE_DIR ${THIRDPARTY_DIR}/opencv)
    set(OpenCV_BINARY_DIR ${THIRDPARTY_DIR}/opencv_build)
    set(OpenCV_INSTALL_DIR ${THIRDPARTY_DIR}/opencv_install)
    set(OpenCV_CONTRIB_DIR ${THIRDPARTY_DIR}/opencv_contrib)

    if(NOT EXISTS ${OpenCV_SOURCE_DIR}/CMakeLists.txt)
        message(STATUS "Cloning OpenCV...")
        execute_process(COMMAND git clone --depth 1 --branch 4.x https://github.com/opencv/opencv.git ${OpenCV_SOURCE_DIR})
    endif()

    if(NOT EXISTS ${OpenCV_CONTRIB_DIR}/modules/tracking)
        message(STATUS "Cloning OpenCV contrib modules...")
        execute_process(COMMAND git clone --depth 1 --branch 4.x https://github.com/opencv/opencv_contrib.git ${OpenCV_CONTRIB_DIR})
    endif()

    ExternalProject_Add(OpenCV_ext
        SOURCE_DIR ${OpenCV_SOURCE_DIR}
        BINARY_DIR ${OpenCV_BINARY_DIR}
        INSTALL_DIR ${OpenCV_INSTALL_DIR}

        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -S ${OpenCV_SOURCE_DIR}
            -B ${OpenCV_BINARY_DIR}
            -DCMAKE_INSTALL_PREFIX=${OpenCV_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=ON
            -DBUILD_LIST=core,imgproc,videoio,tracking,objdetect,imgcodecs,highgui
            -DOPENCV_EXTRA_MODULES_PATH=${OpenCV_CONTRIB_DIR}/modules
            -DWITH_GDAL=OFF
            -DWITH_OPENEXR=OFF
            -DWITH_PROTOBUF=ON
            -DBUILD_PROTOBUF=ON
            -DWITH_OPENCL=ON
            -DWITH_TBB=ON
            -DWITH_EIGEN=OFF
            -DWITH_V4L=ON
            -DWITH_FFMPEG=ON
            -DWITH_GSTREAMER=ON
            -DWITH_MSMF=ON
            -DWITH_AVFOUNDATION=ON
            -DWITH_JPEG=ON
            -DBUILD_JPEG=ON
            -DBUILD_opencv_apps=OFF
            -DBUILD_opencv_calib3d=ON
            -DBUILD_opencv_features2d=ON
            -DBUILD_opencv_flann=ON
            -DBUILD_opencv_ml=ON
            -DBUILD_opencv_photo=ON
            -DBUILD_opencv_stitching=ON
            -DBUILD_opencv_dnn=ON
            -DBUILD_opencv_video=ON
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_EXAMPLES=OFF
            -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON

        BUILD_COMMAND ${CMAKE_COMMAND} --build ${OpenCV_BINARY_DIR} --target install -- -j${OPENCV_BUILD_JOBS}
        BUILD_BYPRODUCTS ${OpenCV_INSTALL_DIR}/lib64/libopencv_core.so
    )
endif()

# === App Sources ===
add_executable(blinkGesture
    main.cpp
    src/eyestrip.cpp
    src/gui.cpp
    src/blink_detector.cpp
    src/command_dispatcher.cpp
)

target_sources(blinkGesture PRIVATE include/gui.hpp)

# === System Dependencies ===
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBEVDEV REQUIRED libevdev)

# === Include Paths ===
target_include_directories(blinkGesture PRIVATE include)

if(USE_SYSTEM_ONNX)
    target_include_directories(blinkGesture PRIVATE ${ONNXRUNTIME_INCLUDE})
else()
    target_include_directories(blinkGesture PRIVATE ${ONNX_INSTALL_DIR}/include)
    target_link_directories(blinkGesture PRIVATE ${ONNX_INSTALL_DIR}/lib)
endif()

if(USE_SYSTEM_OPENCV)
    target_include_directories(blinkGesture PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(blinkGesture PRIVATE ${OpenCV_LIBS})
else()
    target_include_directories(blinkGesture PRIVATE ${OpenCV_INSTALL_DIR}/include/opencv4)
    target_link_directories(blinkGesture PRIVATE ${OpenCV_INSTALL_DIR}/lib64)
    add_dependencies(blinkGesture OpenCV_ext)

    target_link_libraries(blinkGesture PRIVATE
        opencv_core
        opencv_imgproc
        opencv_videoio
        opencv_tracking
        opencv_objdetect
        opencv_imgcodecs
    )
endif()

# === Final Linking ===
if(USE_SYSTEM_ONNX)
    target_link_libraries(blinkGesture PRIVATE ${ONNXRUNTIME_LIB})
else()
    target_link_libraries(blinkGesture PRIVATE onnxruntime)
endif()

target_link_libraries(blinkGesture PRIVATE
    -Wl,--as-needed
    Qt5::Widgets
    ${LIBEVDEV_LIBRARIES}
)
