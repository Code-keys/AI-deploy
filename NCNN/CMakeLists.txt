cmake_minimum_required(VERSION 3.4.1)
 
set(CMAKE_CXX_STANDARD 11)

project( main )

if(MSVC) 
    add_definitions(/wd4018 /wd4244 /wd4996)
endif()
 
# directory of opencv headers
find_library( ncnn PATH C:/opt/ncnn-20211208-windows-vs2019-shared/x64/lib/cmake/ncnn )  
include_directories( C:/opt/ncnn-20211208-windows-vs2019-shared/x64/include )  
link_directories( C:/opt/ncnn-20211208-windows-vs2019-shared/x64/lib )

 # find required opencv
set(OpenCV_DIR  C:/opt/OpenCV/build/x64/vc15/lib)
find_package(OpenCV 3 REQUIRED ) 
include_directories(${OpenCV_INCLUDE_DIRS})  
link_directories(${OpenCV_LIBRARY_DIRS}) 


add_executable(main   main.cpp )
target_link_libraries(main  ncnn  ${OpenCV_LIBS} ) 
 