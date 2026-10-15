#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/matrix_coord.h"


namespace toptlass{
namespace layout {


class RowMajor {

public:
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank,LongIndex>;

private:

    Stride stride_;

public:

    TOPTLASS_HOST_DEVICE
    RowMajor(LongIndex ldm = 0):stride_(ldm){}

    TOPTLASS_HOST_DEVICE
    RowMajor(Stride stride):stride_(stride){}

};


class ColumnMajor {
public:
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, LongIndex>;

private:
  //
  // Data members
  //

  /// Stride data member
  Stride stride_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajor(LongIndex ldm = 0): stride_(ldm) { }
  
  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajor(Stride stride): stride_(stride) { }

};

template <int Interleave>
struct RowMajorInterleaved {
  
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, LongIndex>;

  /// Size of interleaved columns
  static int const kInterleave = Interleave;

private:
  //
  // Data members
  //

  /// Stride data member
  Stride stride_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorInterleaved(LongIndex ldm = 0): stride_(ldm) { }
  
  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorInterleaved(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorInterleaved packed(MatrixCoord const &extent) {
    return RowMajorInterleaved(extent.column() * kInterleave);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (row, column)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord const &coord) const {
    Index row_major = coord.row() / kInterleave;
    Index row_minor = coord.row() % kInterleave;
    return LongIndex(row_major) * LongIndex(stride_[0]) + LongIndex(coord.column()) * kInterleave + row_minor;
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  MatrixCoord inverse(LongIndex offset) const {

    Index row_major = Index(offset / stride_[0]);
    Index residual = Index(offset % stride_[0]);

    Index column = residual / kInterleave;
    Index row_minor =  residual % kInterleave;

    return MatrixCoord(row_major * kInterleave + row_minor, column);
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  typename Stride::Index stride(int idx) const {
    return stride_[idx];
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  typename Stride::Index & stride(int idx) {
    return stride_[idx];
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(MatrixCoord const &extent) const {
    return (extent.row() + kInterleave - 1) / kInterleave * stride_[0];
  }
};


}
}