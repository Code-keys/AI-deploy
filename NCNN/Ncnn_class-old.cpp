#ifndef DarkNCNNet_H
#define DarkNCNNet_H 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <chrono>

#include "ncnn/net.h"  

namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}
typedef struct {
    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;
typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;
inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}


class DarkNCNNet {
public:
    DarkNCNNet(const char* param, const char* bin, bool useGPU, int ins = 640) :input_size(ins) {
        hasGPU = 0;//  ncnn::get_gpu_count() > 0;
        toUseGPU = hasGPU && useGPU;

        Net = new ncnn::Net();
        // opt 需要在加载前设置
        Net->opt.use_vulkan_compute = toUseGPU;  // gpu
        Net->opt.use_fp16_arithmetic = true;  // fp16运算加速
        Net->load_param(param);
        Net->load_model(bin);
    };

    ~DarkNCNNet() {
        Net->clear();
        delete Net;
    };

    std::vector<BoxInfo> detect(ncnn::Mat in_net, float confshold) {

        std::vector<BoxInfo> result;
        ncnn::Mat blob; //out tensor

        float norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f }, mean[3] = { 0, 0, 0 };
        in_net.substract_mean_normalize(mean, norm);

        auto ex = Net->create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(4);
        if (toUseGPU) {  // 消除提示
            /* ex.set_vulkan_compute(toUseGPU); */
        }
        ex.input(0, in_net);
        ex.extract("output", blob);
        auto boxes = decode_infer(blob, { (int)in_net.w, (int)in_net.h }, input_size, num_class, confshold);
        result.insert(result.begin(), boxes.begin(), boxes.end());
        //    nms(result, nms_threshold); 

        for (int i = 0; i < 1000; i++) {
            ex.input("data", in_net);
            ex.extract("output", blob);
        }
        return result;
    };

    float detect_cv(cv::Mat& image/* BGR */, float threshold, float nms_threshold) {

        auto start = std::chrono::high_resolution_clock::now();
        ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(
            (image).data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, input_size, input_size);
        std::vector<BoxInfo> Boxes = detect(in_net, nms_threshold);
        for (auto& box : Boxes)
            if (box.score > threshold) {
               // cv::Rectangle(image, (box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1), cv::Scalar(0x27, 0xC1, 0x36), 1);
            } 
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return duration.count();
    }

    std::vector<std::string> labels{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush" };
private:
    static std::vector<BoxInfo>
        decode_infer(ncnn::Mat& data, const yolocv::YoloSize& frame_size, int net_size, int num_classes, float confshold = 0.1) {
        std::vector<BoxInfo> result;
        for (int i = 0; i < data.h; i++) {
            BoxInfo box;
            const float* values = data.row(i);
            if (values[1] <= confshold) continue;
            box.label = values[0] - 1;
            box.score = values[1];
            box.x1 = values[2] * (float)frame_size.width;
            box.y1 = values[3] * (float)frame_size.height;
            box.x2 = values[4] * (float)frame_size.width;
            box.y2 = values[5] * (float)frame_size.height;
            result.push_back(box);
        }
        return result;
    };

    //    static void nms(std::vector<BoxInfo>& result,float nms_threshold);
    ncnn::Net* Net;
    int input_size;
    int num_class = 80;
public:
    static DarkNCNNet* detector;
    static bool hasGPU;
    static bool toUseGPU;
};

bool DarkNCNNet::hasGPU = true;
bool DarkNCNNet::toUseGPU = true;
DarkNCNNet* DarkNCNNet::detector = nullptr;

#endif //DarkNCNNet_H
 