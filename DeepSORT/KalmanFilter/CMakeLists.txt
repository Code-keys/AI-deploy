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
    ${CMAKE_CURRENT_SOURCE_DIR}/../DeepAppearanceDescriptor/
    ${CMAKE_CURRENT_SOURCE_DIR}/../MunkresAssignment/ 
)
link_directories( )

set(Eigen3_path /home/nvidia/CX/Tracks/eigen-3.3.9/cmake) 
find_package(Eigen3 PATHS ${Eigen3_path} REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_library( KCF
# add_executable( KCF
    kalmanfilter.cpp

    linear_assignment.cpp

    nn_matching.cpp

    track.cpp

    tracker.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/../DeepAppearanceDescriptor/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../MunkresAssignment/hungarianoper.cpp
)  
target_link_libraries(KCF ${OpenCV_LIBS}) 

add_definitions(-O2 -pthread) 