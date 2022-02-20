#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "../DeepAppearanceDescriptor/model.h"
#include "../DeepAppearanceDescriptor/dataType.h"
typedef unsigned char uint8;

class TF_FeatureTensor
{
public:
	static TF_FeatureTensor* getInstance();
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);

private:
	TF_FeatureTensor();
	TF_FeatureTensor(const TF_FeatureTensor&);
	TF_FeatureTensor& operator = (const TF_FeatureTensor&);
	static TF_FeatureTensor* instance;
	bool init();
	~TF_FeatureTensor();

	void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

	int feature_dim;
	tensorflow::Session* session;
	std::vector<tensorflow::Tensor> output_tensors;
	std::vector<tensorflow::string> outnames;
	tensorflow::string input_layer;
public:
	void test();
};
