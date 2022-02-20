#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <string>
#include <opencv2/imgproc/imgproc_c.h>


extern int test(float confidenceThreshold = 0.25, float nms = 0.45);
