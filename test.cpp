#include "infeRt_class.h"
#include "TorchScpt_class.h"
#include "yolo_class.hpp"
#include <vector>


std::string video_path = "/home/nvidia/CX/VisDrone/uav_CrossRoad.avi" ;

cv::Mat img ; 
cv::VideoCapture cap(video_path);


int TestTensorrt(char* pt){ 
    cap >> img;
    cap >> img; 
    std::string engine = pt ;

    auto model = new TensorRt::Detector( engine );

    std::vector<float> t;
    float tt;
    while(1){
        cap >> img;
        if ( img.empty() )  break;
        float tt = model->predict_cv( img, 0.25, 0.45 );    
        t.push_back( tt );
        std::cout << "the frame cost time: " << tt << " ms\n" ; 
        // cv::imshow("", img); 
        // cv::waitKey(0); 
    }

    std::cout << "Infer done\n" ;
    tt = .0f;
    for(int i=0; i < t.size(); i++)  if (i >= 30) tt += t[i]; 
    tt /= t.size()-30 ;
    std::cout << tt  << std::endl;
    return 0;
 
}; 

int TestLibTorrch(char* pt){
    cap >> img;
    cap >> img; 
    // std::string pt = "/home/nvidia/CX/yolov5-cfg/weights/dw1_best_FP32.torchscript";
    // std::string pt = "/home/nvidia/CX/yolov5-cfg/weights/dw2_best_FP32.torchscript";
    // std::string pt = "/home/nvidia/CX/yolov5-cfg/weights/dw4_best_FP32.torchscript";
    // std::string pt = "/home/nvidia/CX/yolov5-cfg/weights/conv3113-best_FP32.torchscript";
    auto detector = new TorchScpt::Detector( pt ,"" ,0);
    
    std::vector<float> t;
    float tt;

    while(1){
        cap >> img;
        if ( img.empty() )  break ;
        tt = detector->predict_cv( img, 0.25, 0.45 );   
        // cv::imshow("", img); 
        // cv::waitKey();
        t.push_back( tt );
        std::cout << "the frame cost time: " << tt << " ms\n" ; 
    }
    std::cout << "Infer done\n" ;
    tt = .0f;
    for(int i=0; i < t.size(); i++){
        if (i < 7) continue;
        tt += t[i];
    }
    tt /= t.size()-7 ;
    std::cout << tt  << std::endl;
    return 0;
};

int TestDarknet(std::string cfg, std::string weights){ 
    cap >> img;
    cap >> img; 
    // std::string cfg = "/home/nvidia/CX/VisDrone/yolov5sm.cfg";
    // std::string weights= "/home/nvidia/CX/VisDrone/yolov5sm.weights";
// 
    auto detectorq = new Detector( cfg ,weights ,0);

    std::vector<float> t;
    float tt;

    while(1){
        cap >> img;
        if ( img.empty() )  break ;
        tt = detectorq->predict_cv( img, 0.25, 0.45 );   
        // cv::imshow("", img); 
        // cv::waitKey();
        t.push_back( tt );
        std::cout << "the frame cost time: " << tt << " ms\n" ; 
    }
    std::cout << "Infer done\n" ;
    tt = .0f;
    for(int i=0; i < t.size(); i++){
        if (i < 7) continue;
        tt += t[i];
    }
    tt /= t.size()-7 ;
    std::cout << tt  << std::endl;
    return 0;

}

int main(int argc,char *argv[]){ 
    std::string type =  argv[1];
    if( type == "1" ) TestTensorrt( argv[2] ); 
    else if ( type == "2" ) TestLibTorrch(argv[2]);
    else if ( type == "3" ) TestDarknet(argv[2], argv[3]);  
    
    return 0;
}

// "/home/nvidia/CX/VisDrone/yolov5sm_fp16.engine"