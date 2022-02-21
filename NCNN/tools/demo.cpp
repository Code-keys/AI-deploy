  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 

#include "ncnn/net.h" 

#include <chrono>
#include <iostream>

static ncnn::Net *net = new ncnn::Net(     ); 
 
int main()
{    //加载模型
    net->load_param("../ncnn-opt.param");
    net->load_model("../ncnn-opt.bin");
	//使用opencv以灰度图读取图片 

    cv::Mat pixels = cv::imread("C:/Users/CX/Pictures/flower.jpeg");
    //获取图片的宽  //获取图;片的高  
    int cols = pixels.cols;
    int rows = pixels.rows; 
    
    //将OpenCV的图片转为ncnn格式的图片,并且将图片缩放到60×60之间
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(pixels.data, ncnn::Mat::PIXEL_BGR, cols, rows, 1536, 768);
    ncnn::Mat feat;

    float mean[1] = { 128.f };
    float norm[1] = { 1/128.f };
	//对图片进行归一化,将像素归一化到-1~1之间
    in.substract_mean_normalize(mean, norm); 

	//定义模型的网络 
	 
    ncnn::Extractor ex = net->create_extractor();
    ex.set_light_mode(true);
	//设置线程个数
    ex.set_num_threads(4); 

	//将图片放入到网络中,进行前向推理
    ex.input("data", in);

	//获取网络的输出结果
    ex.extract("output", feat);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<1000; i++){ 
        ex.input("data", in); 
        ex.extract("output", feat);
    } 
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "post-process takes : " << duration.count() << " ms " << std::endl;
    
    std::cin >> cols;
    return 0;
} 