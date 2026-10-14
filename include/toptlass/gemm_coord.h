#pragma once

#include "toptlass/coord.h"

namespace toptlass{
namespace gemm{

template <
    int M = 1,
    int N = 1,
    int K = 1
>
struct GemmShape {
    static int const kM = M;
    static int const kN = N;
    static int const kK = k;

    static int const kMN = M*N;
    static int const kMK = M*K;
    static int const kKN = N*K;
    static int const kMNK = M*N*K;

    static int const kCount = kMNK;

    TOPTLASS_HOST_DEVICE
    static Coord<3> toCoord(){
        return make_Coord(kM,kN,kK);
    }
};

template <
    typename Shape
>
using GemmShapeTranspose = GemmShape<Shape::KN,Shape::kM,Shape::kK>;

struct GemmCoord : public Coord<3,int>{

    typedef int Index;

    typeedf Coord<3,Index> Base;

    static int const kM = 0;

    static int const kN = 1;
    
    static int const kK = 2;

    TOPTLASS_HOST_DEVICE
    GemmCoord(){}

    TOPTLASS_HOST_DEVICE
    GemmCoord(Coord<3,Index> const& coord): Base(make_Coord(coord[0],coord[1],coord[2])){}

    TOPTLASS_HOST_DEVICE
    GemmCoord(Index m,Index n,Index k): Base(make_Coord(m,n,k)){}
};

}
}