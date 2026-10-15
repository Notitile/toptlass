#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/array.h"
#include "toptlass/numeric_types.h"
#include "toptlass/matrix_shape.h"
#include "toptlass/gemm/gemm.h"
#include "toptlass/gemm/warp/mma.h"

#include "toptlass/gemm/thread/mma.h"

#include "toptlass/gemm/warp/mma_simt_tile_iterator.h"
#include "toptlass/gemm/warp/mma_simt_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace toptlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK = 1,
  /// Complex transformation on operand A
  ComplexTransform TransformA = ComplexTransform::kNone,
  /// Complex transformation on operand B
  ComplexTransform TransformB = ComplexTransform::kNone,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaSimt {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassSimt;

  /// Hard-coded for now
  using ArchTag = arch::Gcu300;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = TransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = TransformB;

  /// Layout of threads
  using ThreadLayoutA = typename std::conditional<  std::is_same< layout::ColumnMajorInterleaved<4>, LayoutA >::value,
                  layout::ColumnMajor,
                  typename std::conditional < std::is_same< layout::RowMajorInterleaved<4>, LayoutA >::value,
                      layout::RowMajor,
                      LayoutA>::type
                 >::type;
  
  using ThreadLayoutB = typename std::conditional< std::is_same< layout::ColumnMajorInterleaved<4>, LayoutB >::value,
                  layout::ColumnMajor,
                  typename std::conditional < std::is_same< layout::RowMajorInterleaved<4>, LayoutB >::value,
                      layout::RowMajor,
                      LayoutB>::type
                 >::type;

  static constexpr bool use_dp4a = ( std::is_same< layout::ColumnMajorInterleaved<4>, LayoutA>::value || 
                                     std::is_same< layout::RowMajorInterleaved<4>, LayoutA >::value) && 
                                     std::is_same< ElementA, int8_t >::value && 
                                     std::is_same< ElementB, int8_t >::value;

  using dp4a_type = typename std::conditional< use_dp4a , int8_t, bool >::type;

  /// Thread-level matrix multiply accumulate operator
  using ThreadMma = thread::Mma<
    GemmShape<
      Shape::kM / Policy::WarpShape::kRow,
      Shape::kN / Policy::WarpShape::kColumn,
      Policy::LaneMmaShape::kK>,
    ElementA,
    ThreadLayoutA,
    ElementB,
    ThreadLayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAdd,
    dp4a_type
  >;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;
  
  /// Shape of the underlying instruction
  using InstructionShape = GemmShape<1,1,use_dp4a ? 4 : 1>;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA = FragmentA;

  /// Iterates over the B operand in memory
  using IteratorB = MmaSimtTileIterator<
    MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentB = FragmentB;

  /// Iterates over the C operand in memory
  using IteratorC = MmaSimtTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    Operand::kC,
    ElementC,
    LayoutC,
    Policy
  >;

  /// Storage for C tile
  using FragmentC = typename ThreadMma::FragmentC;

public:

  //
  // Methods
  //

  /// Ctor
  TOPTLASS_DEVICE
  MmaSimt() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  TOPTLASS_DEVICE
  void operator()(
    FragmentC &d, 
    FragmentA a, 
    FragmentB b, 
    FragmentC const &c, int group_idx = 0) const {

    ThreadMma mma;

    if (kTransformA == ComplexTransform::kConjugate) {
      conjugate<FragmentA> conj_a;
      a = conj_a(a);
    }

    if (kTransformB == ComplexTransform::kConjugate) {
      conjugate<FragmentB> conj_b;
      b = conj_b(b);
    }

    mma(d, a, b, c);
  }

  /// Transform the mma operands to the required types
  TOPTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {
    dst_A = A;
    dst_B = B;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace toptlass
