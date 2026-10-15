#pragma once

#include "toptlass/pitch_linear_coord.h"

namespace toptlass {
namespace layout {

template <int Contiguous,int Strided>
    using PitchLinearShape = toptlass::PitchLinearShape<Contiguous,Strided>;
    using PitchLinearCoord = toptlass::PitchLinearCoord;

}
}