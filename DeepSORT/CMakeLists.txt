cmake_minimum_required(VERSION 3.6)

project(KCF)
  
#SET(CMAKE_BUILD_TYPE "Release")W
add_definitions(-DUNICODE -D_UNICODE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3 -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
  
# add all head and lib
include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
link_directories( )

set(Eigen3_path /home/nvidia/CX/Tracks/eigen-3.3.9/cmake) 
find_package(Eigen3 PATHS ${Eigen3_path} REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_library( KCF
# add_executable( KCF
    main.cpp
    
    kalmanfilter.cpp

    linear_assignment.cpp

    nn_matching.cpp

    track.cpp

    tracker.cpp

    DeepAppearanceDescriptor/model.cpp
    MunkresAssignment/hungarianoper.cpp
)  
target_link_libraries(KCF ${OpenCV_LIBS}) 

add_definitions(-O2 -pthread) 







#############################################################
set (  Qt_pro
"
TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt



INCLUDEPATH += /usr/local/include \
/usr/local/include/opencv4 \
/usr/local/include/opencv4/opencv2 \
                    /home/santiago/code/tensorflow \
                    /home/santiago/code/tensorflow/bazel-genfiles \
                    /home/santiago/code/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include \
                    /home/santiago/code/tensorflow/tensorflow/contrib/makefile/gen/host_obj \
                    /home/santiago/code/tensorflow/tensorflow/contrib/makefile/gen/proto \
                    /home/santiago/code/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public  \
                    /home/santiago/code/tensorflow/tensorflow/contrib/makefile/downloads/eigen  \
                    /home/santiago/code/tensorflow/bazel-out/local_linux-py3-opt/genfiles  \
                    /home/santiago/code/tensorflow/tensorflow/contrib/makefile/downloads/absl \


LIBS += /usr/local/lib/libopencv_calib3d.so \
         /usr/local/lib/libopencv_core.so    \
        /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_imgcodecs.so\
        /usr/local/lib/libopencv_objdetect.so\
        /usr/local/lib/libopencv_photo.so \
        /usr/local/lib/libopencv_dnn.so \
        /usr/local/lib/libopencv_features2d.so \
        /usr/local/lib/libopencv_stitching.so \
        /usr/local/lib/libopencv_flann.so\
        /usr/local/lib/libopencv_videoio.so \
        /usr/local/lib/libopencv_video.so\
        /usr/local/lib/libopencv_ml.so \
    /home/santiago/code/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so \
    /home/santiago/code/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so \



SOURCES += \
    DeepAppearanceDescriptor/FeatureTensor.cpp \
    DeepAppearanceDescriptor/model.cpp \
    KalmanFilter/kalmanfilter.cpp \
    KalmanFilter/linear_assignment.cpp \
    KalmanFilter/nn_matching.cpp \
    KalmanFilter/track.cpp \
    KalmanFilter/tracker.cpp \
    MunkresAssignment/munkres/munkres.cpp \
    MunkresAssignment/hungarianoper.cpp \
    main.cpp 

HEADERS += \
    DeepAppearanceDescriptor/dataType.h \
    DeepAppearanceDescriptor/FeatureTensor.h \
    DeepAppearanceDescriptor/model.h \
    KalmanFilter/kalmanfilter.h \
    KalmanFilter/linear_assignment.h \
    KalmanFilter/nn_matching.h \
    KalmanFilter/track.h \
    KalmanFilter/tracker.h \
    MunkresAssignment/munkres/matrix.h \
    MunkresAssignment/munkres/munkres.h \
    MunkresAssignment/hungarianoper.h
")