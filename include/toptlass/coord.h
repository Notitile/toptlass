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

    template <int R,typename I,typename L>
    TOPTLASS_HOST_DEVICE
    Coord(Coord<R,I,L> other){
        for (int i = 0;i<kRank;++i){
            idx[i] = other[i];
        }
    }

    template <int Slice>
    TOPTLASS_HOST_DEVICE
    Coord<Slice, Index, LongIndex> slice(int start = 0, Index identity = 0) const {
        Coord<Slice, Index, LongIndex> result;
        for (int i = 0; i < Slice; ++i) {
        if (i + start < kRank) {
            result[i] = idx[i + start];
        }
        else {
            result[i] = identity;
        }
        }
        return result;
    }

    /// Returns the index of the dimension with least value
    TOPTLASS_HOST_DEVICE
    int min_dim_index() const {
        int i = 0;
        for (int j = 1; j < kRank; ++j) {
        if (idx[j] < idx[i]) {
            i = j;
        }
        }
        return i;
    }

    /// Returns the index of the dimension with greatest value
    TOPTLASS_HOST_DEVICE
    int max_dim_index() const {
        int i = 0;
        for (int j = 1; j < kRank; ++j) {
        if (idx[j] > idx[i]) {
            i = j;
        }
        }
        return i;
    }

    /// Returns true if Coord is non-zero.
    TOPTLASS_HOST_DEVICE
    explicit operator bool() const {
        for (int i = 0; i < kRank; ++i) {
        if (idx[i]) {
            return true;
        }
        }
        return false;
    }

    /// Returns true if Coord is uniformly zero.
    TOPTLASS_HOST_DEVICE
    bool operator!() const {
        for (int i = 0; i < kRank; ++i) {
        if (idx[i]) {
            return false;
        }
        }
        return true;
    }

    /// Element-wise addition
    TOPTLASS_HOST_DEVICE
    Coord operator+(Coord const& b) const {
        Coord c;
        for (int i = 0; i < kRank; ++i) {
        c.idx[i] = idx[i] + b.idx[i];
        }
        return c;
    }

    /// Element-wise subtraction
    TOPTLASS_HOST_DEVICE
    Coord operator-(Coord const& b) const {
        Coord c;
        for (int i = 0; i < kRank; ++i) {
        c.idx[i] = idx[i] - b.idx[i];
        }
        return c;
    }

    /// Element-wise multiplication
    TOPTLASS_HOST_DEVICE
    Coord operator*(Coord const& b) const {
        Coord c;
        for (int i = 0; i < kRank; ++i) {
        c.idx[i] = idx[i] * b.idx[i];
        }
        return c;
    }

    /// Element-wise division
    TOPTLASS_HOST_DEVICE
    Coord operator/(Coord const& b) const {
        Coord c;
        for (int i = 0; i < kRank; ++i) {
        c.idx[i] = idx[i] / b.idx[i];
        }
        return c;
    }

    /// In-place addition
    TOPTLASS_HOST_DEVICE
    Coord& operator+=(Coord const& b) {
        for (int i = 0; i < kRank; ++i) {
        idx[i] += b.idx[i];
        }
        return *this;
    }

    /// In-place subtraction
    TOPTLASS_HOST_DEVICE
    Coord& operator-=(Coord const& b) {
        for (int i = 0; i < kRank; ++i) {
        idx[i] -= b.idx[i];
        }
        return *this;
    }

    /// In-place multiplication
    TOPTLASS_HOST_DEVICE
    Coord& operator*=(Coord const& b) {
        for (int i = 0; i < kRank; ++i) {
        idx[i] *= b.idx[i];
        }
        return *this;
    }

    /// In-place division
    TOPTLASS_HOST_DEVICE
    Coord& operator/=(Coord const& b) {
        for (int i = 0; i < kRank; ++i) {
        idx[i] /= b.idx[i];
        }
        return *this;
    }

    /// Member access operator
    TOPTLASS_HOST_DEVICE Index& operator[](int dim) { return idx[dim]; }

    /// Member access operator
    TOPTLASS_HOST_DEVICE Index const& operator[](int dim) const { return idx[dim]; }

    /// Computes the dot product with anotherCoord object
    TOPTLASS_HOST_DEVICE
    LongIndex dot(Coord const& b, LongIndex sum = LongIndex(0)) const {
        for (int i = 0; i < kRank; ++i) {
        sum += idx[i] * b.idx[i];
        }
        return sum;
    }

    /// Gets the index of a given Coord element
    template <int Dim>
    TOPTLASS_HOST_DEVICE Index& at() {
        return idx[Dim];
    }

    /// Access via index; may limit unrolling potential
    TOPTLASS_HOST_DEVICE
    Index& at(int dim) { return idx[dim]; }

    /// Gets the index of a given Coord element
    template <int Dim>
    TOPTLASS_HOST_DEVICE Index const& at() const {
        return idx[Dim];
    }

    /// Access via index; may limit unrolling potential
    TOPTLASS_HOST_DEVICE
    Index const& at(int dim) const { return idx[dim]; }

    /// Determines if two Coord<> objects are equal
    TOPTLASS_HOST_DEVICE
    bool operator==(Coord const& b) const {
        bool equal = true;
        for (int i = 0; equal && i < kRank; ++i) {
        equal = (idx[i] == b.idx[i]);
        }
        return equal;
    }

    /// Not equal
    TOPTLASS_HOST_DEVICE
    bool operator!=(Coord const& b) const { return !(*this == b); }

    /// Clamps a coordinate to a range specified by maximum and minimum values
    TOPTLASS_HOST_DEVICE
    Coord& clamp(Coord const& max, Coord const& min = Coord()) {
        for (int i = 0; i < kRank; ++i) {
        idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
        }
        return *this;
    }

    /// Returns the sum of all elements
    TOPTLASS_HOST_DEVICE
    Index sum() const {
        Index sum_(idx[0]);
        for (int i = 1; i < kRank; ++i) {
        sum_ += idx[i];
        }
        return sum_;
    }

    /// Returns the product of all elements
    TOPTLASS_HOST_DEVICE
    LongIndex product() const {
        LongIndex product_(idx[0]);
        for (int i = 1; i < kRank; ++i) {
        product_ *= idx[i];
        }
        return product_;
    }

    /// Less than operator
    TOPTLASS_HOST_DEVICE
    bool operator<(Coord const &b) const {
        for (int i = 0; i < kRank; ++i) {
        if (!(idx[i] < b[i])) {
            return false;
        }
        }
        return true;
    }

    /// Less than or equals operator
    TOPTLASS_HOST_DEVICE
    bool operator<=(Coord const &b) const {
        for (int i = 0; i < kRank; ++i) {
        if (!(idx[i] <= b[i])) {
            return false;
        }
        }
        return true;
    }

    /// Greater than operator
    TOPTLASS_HOST_DEVICE
    bool operator>(Coord const &b) const {
        return !(*this <= b);
    }

    /// Greater than or equals operator
    TOPTLASS_HOST_DEVICE
    bool operator>=(Coord const &b) const {
        return !(*this < b);
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

template <typename T> 
TOPTLASS_HOST_DEVICE
Coord<2, T> make_Coord(T _0, T _1) {
  T values[2] = {_0, _1};
  return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <typename T> 
TOPTLASS_HOST_DEVICE
Coord<3, T> make_Coord(T _0, T _1, T _2) {
  T values[3] = {_0, _1, _2};
  return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <typename T> 
TOPTLASS_HOST_DEVICE
Coord<4, T> make_Coord(T _0, T _1, T _2, T _3) {
  T values[4] = {_0, _1, _2, _3};
  return Coord<4, T>(values);
}

}