#pragma once

#include "toptlass/cutlass.h"
#include "toptlass/tensor_ref.h"
#include "toptlass/layout/matrix.h"
#include "toptlass/layout/pitch_linear.h"

namespace toptlass{
namespace thransform{
namespace threadblock{

template <
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  int Alignment = sizeof_bits<Element>::value * ThreadMap::kElementsPerAccess / 8
>
class RegularTileIterator;

/// Regular tile iterator specialized for row major 
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  using Underlying = RegularTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    kAlignment
  >;

  using AccessType = typename Underlying::AccessType;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator() { }

  CUTLASS_DEVICE
  RegularTileIterator(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.column(), tile_offset.row()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Regular tile iterator specialized for pitch-linear
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int Alignment
>
class RegularTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank, ThreadMap_, Alignment> {
public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Fragment = Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  using Underlying = RegularTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap
  >;

  using AccessType = typename Underlying::AccessType;

  static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, 
    "Advance rank may only be along the row or column dimensions.");

private:

  Underlying iterator_;

public:

  CUTLASS_DEVICE
  RegularTileIterator() { }

  CUTLASS_DEVICE
  RegularTileIterator(
    TensorRef const &ref, 
    int thread_idx
  ):
    iterator_({ref.data(), ref.stride()}, thread_idx) {

  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, TensorCoord const & tile_offset) {
    iterator_.load_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    iterator_.load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag, TensorCoord const & tile_offset) {
    iterator_.store_with_pointer_offset(frag, {tile_offset.row(), tile_offset.column()});
  }

  /// Stores a fragment
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    iterator_.store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  RegularTileIterator &operator--() {
    --iterator_;
    return *this;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

};
    
}
}
}