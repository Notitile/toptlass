/*! \file
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/tensor_ref.h"
#include "toptlass/layout/matrix.h"
#include "toptlass/gemm/gemm.h"
#include "toptlass/gemm/thread/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace toptlass {
namespace gemm {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles conventional layouts for IDP4A
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_
>
struct Mma<
  Shape_,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int32_t,
  LayoutC_,
  arch::OpMultiplyAdd,
  bool> {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = int8_t;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = layout::RowMajor;

  /// Data type of operand B
  using ElementB = int8_t;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = layout::ColumnMajor;

  /// Element type of operand C
  using ElementC = int32_t;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  //  Use 1x1x4 IDP4A sequence for bulk of computation
  using ArchMmaOperator = arch::Mma<
      gemm::GemmShape<1,1,4>,
      1,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      arch::OpMultiplyAdd>; 

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  TOPTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementC, LayoutC> d(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed({ Shape::kM, Shape::kN }));
    
    // Copy accumulators
    D = C;

    /// Use 1x1x4 IDP4A sequence for bulk of computation
    ArchMmaOperator mma;

    // Compute matrix product
    TOPTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK / ArchMmaOperator::Shape::kK; ++k) {

      TOPTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; ++n) {

        TOPTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {
          MatrixCoord mn(m, n);

          Array<int8_t, 4> const *ptr_A = reinterpret_cast<Array<int8_t, 4> const *>(&A);
          Array<int8_t, 4> const *ptr_B = reinterpret_cast<Array<int8_t, 4> const *>(&B);

          Array<int32_t, 1> tmp = reinterpret_cast<Array<int32_t, 1> &>(d.at(mn));

          mma(
            tmp,
            ptr_A[m * Shape::kK / ArchMmaOperator::Shape::kK + k],
            ptr_B[n * Shape::kK / ArchMmaOperator::Shape::kK + k],
            tmp);

          d.at(mn) = reinterpret_cast<int32_t &>(tmp);
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Gemplate that handles conventional layouts for IDP4A
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_
>
struct Mma<
  Shape_,
  int8_t,
  layout::ColumnMajor,
  int8_t,
  layout::RowMajor,
  int32_t,
  LayoutC_,
  arch::OpMultiplyAdd,
  int8_t> {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = int8_t;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = layout::ColumnMajor;

  /// Data type of operand B
  using ElementB = int8_t;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = layout::RowMajor;

  /// Element type of operand C
  using ElementC = int32_t;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  /// Use 1x1x4 IDP4A sequence for bulk of computation
  using ArchMmaOperator = arch::Mma<
      gemm::GemmShape<1,1,4>,
      1,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      arch::OpMultiplyAdd>; 

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  TOPTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementC, LayoutC> d(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed({ Shape::kM, Shape::kN }));
    
    // Copy accumulators
    D = C;

    /// Underlying matrix multiply operator
    ArchMmaOperator mma;
    
    Array<int8_t, 4> const *ptr_A = reinterpret_cast<Array<int8_t, 4> const *>(&A);
    Array<int8_t, 4> const *ptr_B = reinterpret_cast<Array<int8_t, 4> const *>(&B);

    // Compute matrix product
    TOPTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK / ArchMmaOperator::Shape::kK; ++k) {

      TOPTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; ++n) {

        TOPTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {
          MatrixCoord mn(m, n);

          Array<int32_t, 1> tmp = reinterpret_cast<Array<int32_t, 1> &>(d.at(mn));

          mma(
            tmp,
            ptr_A[m + k * Shape::kM],
            ptr_B[n + k * Shape::kN],
            tmp);

          d.at(mn) = reinterpret_cast<int32_t &>(tmp);
        }
      }
    }
  }
};

} // namespace thread
} // namespace gemm
} // namespace toptlass

/////////////////////////////////////////////////////////////////////////////////////////////////
