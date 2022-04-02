// copyright cx

#ifndef LAYER_SILU_H
#define LAYER_SILU_H

#include "layer.h"

namespace ncnn {

class Silu : public Layer
{
public:
    Silu();

    int forward_inplace(Mat& bottom_top_blob, const Option& opt) const ;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_H
