#pragma once

#include "toptlass/toptlass.h"

namespace toptlass{
namespace gemm{
namespace threadblock{

template <int N=1>
struct GemmIdentityThreadblockSwizzle {

    TOPTLASS_HOST_DEVICE
    GemmIdentityThradblockSwizzle(){ }

    TOPTLASS_HOST_DEVICE
    static GemmCoord get_tiled_shape(
        GemmCoord problem_size,
        GemmCoord tile_size,
        int split_k_slices){
        
        return GemmCoord(
            (problem_size.m()+tile_size.m()-1)/tile_size.m(),
            (problem_size.n()+tile_size.n()-1)/tile_size.n(),
            split_k_slices);
    }
};

}
}
}