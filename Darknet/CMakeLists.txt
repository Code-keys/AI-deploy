cmake_minimum_required(VERSION 3.6)

project(Detector)

# add_definitions(-std=c++14)

# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE Debug) 


#SET(CMAKE_BUILD_TYPE "Release")
add_definitions(-DUNICODE -D_UNICODE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3 -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
 
# Libtorch
set(Torch_DIR C:/opt/libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

# add all head and lib
include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/ 
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/
C:\\Users\\CX\\Desktop\\darknet-yoloV4\\darknet-master\\build\\Debug
)
    
find_package(OpenCV  PATHS  C:\\opt\\OpenCV )
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_executable(Detector 
    ${PROJECT_SOURCE_DIR}/main.cpp  
) 
target_link_libraries(Detector darkd.dll) 
target_link_libraries(Detector darkd.lib) 
target_link_libraries(Detector ${OpenCV_LIBS})
target_link_libraries(Detector ${TORCH_LIBRARIES})
 

add_definitions(-O2 -pthread) 