#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 

#include "../DeepAppearanceDescriptor/model.h"
#include "../DeepAppearanceDescriptor/dataType.h"

// using COCO : yolov4 model stage-1/4
// using detector model stage-1/4

class FeatureTensor
{
public:  
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d){};

};

#ifdef TensorFlow 
#include "ByTensorFlow.h"
#else ifdef(Torch) 

#else

#endif