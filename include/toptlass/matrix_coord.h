/*! \file
    \brief Defines a canonical coordinate for rank=2 matrices offering named indices.
*/
#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/coord.h"

namespace toptlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// MatrixCoord wraps Coord<2, int> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, int> {

public:

  /// Integer-valued index
  using Index = int;

  /// Base type is a Coord of rank=2
  using Base = Coord<2, Index>;

  /// LongIndex type
  using LongIndex = typename Base::LongIndex;

private:

  /// Rows dimension
  static int const kRow = 0;

  /// Columns dimension
  static int const kColumn = 1;

public:

  //
  // Methods
  //

  /// Default ctor
  TOPTLASS_HOST_DEVICE
  MatrixCoord() { }

  /// Constructs from Coord<2>
  TOPTLASS_HOST_DEVICE
  MatrixCoord(Coord<2, Index> const &coord): Base(coord) { }

  /// Helper to construct from a row and column
  TOPTLASS_HOST_DEVICE
  MatrixCoord(Index row, Index column): Base(make_Coord(row, column)) { }

  /// Helper to construct from a row and column, which are LongIndex based
  TOPTLASS_HOST_DEVICE
  MatrixCoord(LongIndex row, LongIndex column): Base(make_Coord(Index(row), Index(column))) { }

  /// Returns the row of the coordinate
  TOPTLASS_HOST_DEVICE
  Index const & row() const { return this->at(kRow); }

  /// Returns the row of the coordinate
  TOPTLASS_HOST_DEVICE
  Index & row() { return this->at(kRow); }

  /// Returns the column of the coordinate
  TOPTLASS_HOST_DEVICE
  Index const & column() const { return this->at(kColumn); }

  /// Returns the column of the coordinate
  TOPTLASS_HOST_DEVICE
  Index & column() { return this->at(kColumn); }

  //
  // Coord operators
  //

  /// Element-wise addition
  TOPTLASS_HOST_DEVICE
  MatrixCoord operator+(Base const& b) const {
    return MatrixCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  TOPTLASS_HOST_DEVICE
  MatrixCoord operator-(Base const& b) const {
    return MatrixCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  TOPTLASS_HOST_DEVICE
  MatrixCoord operator*(Base const& b) const {
    return MatrixCoord(Base::operator*(b));
  }

  /// Element-wise division
  TOPTLASS_HOST_DEVICE
  MatrixCoord operator/(Base const& b) const {
    return MatrixCoord(Base::operator/(b));
  }

  /// In-place addition
  TOPTLASS_HOST_DEVICE
  MatrixCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  TOPTLASS_HOST_DEVICE
  MatrixCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  TOPTLASS_HOST_DEVICE
  MatrixCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  TOPTLASS_HOST_DEVICE
  MatrixCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
