cmake_minimum_required(VERSION 3.6)

project(Detector)

# add_definitions(-std=c++14)

# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE Debug) 


#SET(CMAKE_BUILD_TYPE "Release")W
add_definitions(-DUNICODE -D_UNICODE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3 -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
 
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)
# libtorch
set(Torch_DIR /home/nvidia/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

# add all head and lib
include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/Darknet
    ${CMAKE_CURRENT_SOURCE_DIR}/TensorRt
    ${CMAKE_CURRENT_SOURCE_DIR}/TensorRt/Plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/TorchScpt
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/build
    ${CMAKE_CURRENT_SOURCE_DIR}/TensorRt
    ${CMAKE_CURRENT_SOURCE_DIR}/TorchScpt
)

cuda_add_library(myplugins SHARED  
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/mish.h
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/mish.cu 
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/yololayer.h 
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/yololayer.cu 
    # ${PROJECT_SOURCE_DIR}/calibrator.cpp
)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_executable(Detector 
    ${PROJECT_SOURCE_DIR}/test.cpp 
    ${PROJECT_SOURCE_DIR}/TorchScpt/TorchScpt_class.cpp
)
target_link_libraries(Detector nvinfer)
target_link_libraries(Detector cudart)
target_link_libraries(Detector libdarknet.so)
target_link_libraries(Detector myplugins)
target_link_libraries(Detector ${OpenCV_LIBS})
target_link_libraries(Detector ${TORCH_LIBRARIES})
 

add_definitions(-O2 -pthread) 