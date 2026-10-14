#pragma once


#include <cstdint>

namespace toptlass{

template <
    int Rank_,
    typename Index_ = int,
    typename LongIndex_ = int64_t
>
struct Coord{

public:

    static int const kRank = Rank_;

    using Index = Index_;

    using LongIndex = LongIndex_;

private:

    Index idx[kRank];

public:

    TOPTLASS_HOST_DEVICE
    explicit Coord(Index value = Index(0)){
        for (int i = 0; i<kRank;++i){
            idx[i] = value;
        }
    }

    TOPTLASS_HOST_DEVICE
    Coord(Index const (&_idx)[kRank]){
        for (int i = 0; i<kRank;++i){
            idx[i] = _idx[i];
        }
    }

    template <int R,typename I,typname L>
    TOPTLASS_HOST_DEVICE
    Coord(Coord<R,I,L> other){
        for (int i = 0;i<kRank;++i){
            idx[i] = other[i];
        }
    }

};

} // namespace toptlass

namespace toptlass{
    
template <int Rank,typename Index>
TOPTLASS_HOST_DEVICE
Coord<Rank,Index> operator*(Index s,Coord<Rank,Index> coord){
    TOPTLASS_PRAGMA_UNROLL
    for (int i = 0; i<Rank;++i){
        coord[i] *= s;
    }
    return coord;
}


template <typename T>
TOPTLASS_HOST_DEVICE
Coord<1,T> make_Coord(T _0){
    T values[1] = {_0};
    return Coord<1,T>(values);
}

}