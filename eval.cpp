#include "infeRt_class.h"
#include "TorchScpt_class.h"
#include "yolo_class.hpp"

#include <vector>
#include <iostream>
#include <fstream>

#include <stdlib.h>  // 新进程 python eval.py

#include "json.hpp"
using namespace nlohmann; 
using json = nlohmann::json;
 
#include <dirent.h>
static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

int Eval(const char* p_dir_name, const char* pt, const char* cfg=""){

    std::vector<std::string> file_names;
    read_files_in_dir( p_dir_name, file_names);

    std::ofstream o("/home/nvidia/CX/VisDrone/pred.json");//, std::ofstream::app);
    json jCOCO;

    std::string wts = pt ; 
    
    // if( wts.find(".engine") == wts.npos ){
        auto model = new TensorRt::Detector( wts ); 
    // }else if( wts.find(".torch") == wts.npos ){
    //     auto model = new TorchScpt::Detector( wts ,"" ,0);
    // }else if( wts.find(".weights") == wts.npos ){
    //     auto model = new Detector( cfg ,wts ,0);
    // }else{
    //     auto model = 0 ;   return 0;
    // }
 

    for( auto basename : file_names){
        std::string sp_dir_name(p_dir_name) ;

        sp_dir_name += "/";
        sp_dir_name += basename ;  
        cv::Mat img = cv::imread( sp_dir_name ) ;  

        std::vector< std::vector<int> > ret;
        // #ifdef model
        ret =  model->predict(img, 0.001, 0.5);
        // #endif
        if (ret.size() == 0) std::cout << "                                                                 no box det \n" ;
        for(auto box :ret){
            json box_map;
            box_map["image_id"]= basename.substr( 0, basename.length()-4 ) ;
            box_map["category_id"]= box[0] ;
            box_map["score"]= box[1] / 1000.0 ; 
            box_map["bbox"]=  { box[2], box[3], box[4], box[5]};   
            jCOCO.push_back(box_map);
        }
        std::cout << "Processing img: " << basename << std::endl;
    }
    o << std::setw(4) << jCOCO << std::endl;  // write prettified JSON to another file
    return 1;
}

// auto detectorrt = new TensorRt::Detector( engine ); 
// auto detectorts = new TorchScpt::Detector( pt ,"" ,0); 
// auto detectord = new Detector( cfg ,weights ,0);

int main(int argc,char *argv[]){  

    Eval( "/home/nvidia/CX/VisDrone/org_val/" , argv[1] , argv[2]); 
    
    system( "python /home/nvidia/CX/yolov5-cfg/QT/include/eval.py"  );

    return 0;
}

// "/home/nvidia/CX/VisDrone/yolov5sm_fp16.engine"