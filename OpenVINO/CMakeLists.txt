cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)

project(yolox_openvino_demo)


set(OpenCV_DIR C:/opt/Openvino-2021.4.2/openvino_2021/opencv/cmake)
find_package(OpenCV REQUIRED )

set(InferenceEngine_DIR C:/opt/Openvino-2021.4.2/openvino_2021.4.752/deployment_tools/inference_engine/share)
find_package(InferenceEngine REQUIRED ) 

set(ngraph_DIR C:/opt/Openvino-2021.4.2/openvino_2021.4.752/deployment_tools/ngraph/cmake  )
find_package(ngraph  REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(yolox_openvino yolox.cpp)

target_link_libraries(
     yolox_openvino
    ${InferenceEngine_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${OpenCV_LIBS} 
)