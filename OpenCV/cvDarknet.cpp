#include "cvDarknet.h"

int test(float confidenceThreshold, float nms) {

	cv::String modelConfiguration = "C:/Users/CX/Desktop/yolov5/model.cfg";
	cv::String modelBinary = "C:/Users/CX/Desktop/yolov5/weights/best.weights";
	auto frame = cv::imread("C:/Users/CX/iCloudDrive/c++/3d/CL/water\ wa.jpeg");
	 
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary);
	if (net.empty())
	{
		printf("Could not load net...\n");
		return 1;
	}


	std::vector<std::string> classNamesVec;
	std::ifstream classNamesFile("D:/vcprojects/images/dnn/yolov2-tiny-voc/voc.names");
	if (classNamesFile.is_open())
	{
		std::string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}else {
		for (int i = 0; i < 20; i++) classNamesVec.push_back("1");
	}

	// 加载图像
	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false);
	net.setInput(inputBlob, "data");
	 
	// 检测
	cv::Mat detectionMat = net.forward("detection_out");
	std::vector<double> layersTimings;
	double freq = cv::getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	std::ostringstream ss;
	ss << "detection time: " << time << " ms";
	cv::putText(frame, ss.str(), cv::Point(20, 20), 0, 0.5, cv::Scalar(0, 0, 255));

	// 输出结果
	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
		size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
		if (confidence > confidenceThreshold)
		{
			float x = detectionMat.at<float>(i, 0);
			float y = detectionMat.at<float>(i, 1);
			float width = detectionMat.at<float>(i, 2);
			float height = detectionMat.at<float>(i, 3);
			int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
			int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
			int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
			int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
			cv::Rect object(xLeftBottom, yLeftBottom,
				xRightTop - xLeftBottom,
				yRightTop - yLeftBottom);
			cv::rectangle(frame, object, cv::Scalar(0, 0, 255), 2, 8);
			if (objectClass < classNamesVec.size())
			{
				ss.str("");
				ss << confidence;
				cv::String conf(ss.str());
				cv::String label = cv::String(classNamesVec[objectClass]) + ": " + conf;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom),
					cv::Size(labelSize.width, labelSize.height + baseLine)),
					cv::Scalar(255, 255, 255), CV_FILLED);
				cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom + labelSize.height),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}
	}
	cv::imshow("YOLO-Detections", frame);
	cv::waitKey(0);
}