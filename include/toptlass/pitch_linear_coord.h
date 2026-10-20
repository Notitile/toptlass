/*! \file
    \brief Defines layout functions used by TensorRef and derived classes for pitch-linear memory.
*/
#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/coord.h"

namespace toptlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template defining a shape used by pitch-linear operators
template <
  int Contiguous,
  int Strided
>
struct PitchLinearShape {
  static int const kContiguous = Contiguous;
  static int const kStrided = Strided;
  static int const kCount = Contiguous * Strided;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Coordinate in pitch-linear space
struct PitchLinearCoord : public Coord<2, int> {
public:

  /// Integer-valued index
  using Index = int;

  /// Base type is a Coord of rank=2
  using Base = Coord<2, Index>;

  /// Long integer type
  using LongIndex = typename Base::LongIndex;

private:

  /// Rows dimension
  static int const kContiguous = 0;

  /// Columns dimension
  static int const kStrided = 1;

public:

  //
  // Methods
  //

  /// Default ctor
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord() { }

  /// Constructs from Coord<2>
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord(Coord<2, Index> const &coord): Base(coord) { }

  /// Helper to construct from a row and column
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord(Index contiguous_, Index strided_): Base(make_Coord(contiguous_, strided_)) { }

  /// Helper to construct from a row and column based on LongIndex
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord(LongIndex contiguous_, LongIndex strided_)
    : Base(make_Coord(Index(contiguous_), Index(strided_))) { }

  /// Returns the contiguous dimension
  TOPTLASS_HOST_DEVICE
  Index const & contiguous() const { return this->at(kContiguous); }

  /// Returns the contiguous dimension
  TOPTLASS_HOST_DEVICE
  Index & contiguous() { return this->at(kContiguous); }

  /// Returns the column of the coordinate
  TOPTLASS_HOST_DEVICE
  Index const & strided() const { return this->at(kStrided); }

  /// Returns the column of the coordinate
  TOPTLASS_HOST_DEVICE
  Index & strided() { return this->at(kStrided); }

  //
  // Coord operators
  //

  /// Element-wise addition
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord operator+(Base const& b) const {
    return PitchLinearCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord operator-(Base const& b) const {
    return PitchLinearCoord(Base::operator-(b));
  }

  TOPTLASS_HOST_DEVICE
  PitchLinearCoord operator-() const {
    return PitchLinearCoord(-at(0), -at(1));
  }

  /// Element-wise multiplication
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord operator*(Base const& b) const {
    return PitchLinearCoord(Base::operator*(b));
  }

  /// Element-wise division
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord operator/(Base const& b) const {
    return PitchLinearCoord(Base::operator/(b));
  }

  /// In-place addition
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  TOPTLASS_HOST_DEVICE
  PitchLinearCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace toptlass

