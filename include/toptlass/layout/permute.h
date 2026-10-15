#pragma once


#include "toptlass/matrix_coord.h"
#include "toptlass/layout/pitch_linear.h"
#include "toptlass/toptlass.h"

namespace toptlass {
namespace layout {

class PermuteBase {
    using Index = int32_t;
    using LongIndex = int64_t;
};

class NoPermute : public PermuteBase {
public:
    TOPTLASS_HOST_DEVICE
    NoPermute(MatrixCoord extent,Index Stride) {};

    TOPTLASS_HOST_DEVICE
    NoPermute(PitchLinearCoord extent,Index stride){}

};    

}
}