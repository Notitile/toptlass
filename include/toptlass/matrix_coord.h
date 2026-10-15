#pragma once

#include "toptlass/coord.h"
#include "toptlass/toptlass.h"

namespace toptlass {

struct MatrixCoord : public Coord<2,int> {

public: 
    using Index = int;
    using Base = Coord<2,Index>;
    using LongIndex = typename Base::LongIndex;

private:
    static int const kRow = 0;
    static int const kColumn = 1;

public:

    TOPTLASS_HOST_DEVICE
    MatrixCoord(){}

    TOPTLASS_HOST_DEVICE
    MatrixCoord(Coord<2,Index> const &coord):Base(coord) { }

    TOPTLASS_HOST_DEVICE
    MatrixCoord(Index row,Index column):Base(make_Coord(row,column)){}

    TOPTLASS_HOST_DEVICE
    Index const & row() const (return this->at(kRow);)

    TOPTLASS_HOST_DEVICE
    Index & row() {return this->at(kRow);}
    
    TOPTLASS_HOST_DEVICE
    Index & column() {return this->at(kColumn);}
};

}