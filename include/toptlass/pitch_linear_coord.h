#pragma once

#include "toptlass/coord.h"

namespace toptlass {

template <
    int Contiguous,
    int Strided
    >
struct PitchLinearShape {
    static int const kContiguous = Contiguous;
    static int const kStrided = Strided;
    static int const kCount = Contiguous * Strided;
};

struct PitchLinearCoord : public Coord<2,int> {

public:
    using Index = int;
    using Base = Coord<2,int>;
    using LongIndex = typename Base::LongIndex;

private:
    TOPTLASS_HOST_DEVICE
    PitchLinearCoord() {}

    TOPTLASS_HOST_DEVICE
    PitchLinearCoord(Coord<2,Index> const &coord):Base(coord) {}

    TOPTLASS_HOST_DEVICE
    PitchLinearCoord<Index contiguous_, Index strided_):Base(make_Coord(contiguous_,strided_)){}

    
};

}