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
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_executable( KCF 
    TrackTest4.cpp
    ffttools.hpp
    fhog.cpp
    fhog.hpp
    kcftracker.cpp
    kcftracker.hpp
    labdata.hpp
    recttools.hpp
    runtracker.cpp
    tracker.h 
)  
target_link_libraries(KCF ${OpenCV_LIBS}) 
  
add_definitions(-O2 -pthread) 