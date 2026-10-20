#pragma once

#include "toptlass/gemm_coord.h"
#include "toptlass/gemm/warp/mma.h"
#include "toptlass/arch/mma.h"
#include "toptlass/layout/matrix.h"
#include "toptlass/gemm/threadblock/default_mma_core.h"

namespace toptlass {
namespace gemm {
namespace threadblock{


namespace detail {

// convert a WarpShape which is the whole tile of elements into warp num threads.
// The goal is for each thread's tile of elements to be as square as possible
// for performance (4x4 will be faster than 2x8).
template<typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}

/// Computes padding in shared memory to perform efficient transpose without bank conflicts.
constexpr int simt_transpose_padding(int threads, int crosswise, int size_in_bits) {
  return (size_in_bits >= 32 ?
      threads / crosswise / (size_in_bits / 32) :
      threads / crosswise * (32 / size_in_bits)
  );
}

}



template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
    using Shape = Shape_;

    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using WarpCount = GemmShape<
                        Shape::kM / WarpShape::kM,
                        Shape::kN / WarpShape::kN,
                        PartitionsK>;

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;    
    
    static int const kThreads = WarpCount::kCount * kWarpSize;
  
    static int const kElementsPerAccess = 1;

    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;
    
    /// ThreadMap of iterator A
    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kK, Shape::kM>,
        kThreads,
        kElementsPerAccess
    >;

    /// Transpose the ThreadMap of iterator A
    using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    /// Shared memory iterator to A operand
    using SmemIteratorA = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, 
        ElementA, 
        SmemLayoutA,
        1,
        SmemThreadMapA // was IteratorThreadMapA
    >;

    /// ThreadMap of iterator B
    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kK, Shape::kN>,
        kThreads,
        kElementsPerAccess
    >;

    /// Transpose the ThreadMap of iterator A
    using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

    /// Shared memory iterator to B operand
    using SmemIteratorB = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kK, Shape::kN>, 
        ElementB, 
        SmemLayoutB,
        0,
        SmemThreadMapB // was IteratorThreadMapA
    >;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level op
    static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
        "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = toptlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = toptlass::const_min(numElementsB, ThreadTileN);

    static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
    static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    static_assert(!(kPaddingM % LaneM) && !(kPaddingN % LaneN),
                    "Padding must be divisible by Lane");

    // these should have max of thread tile also
    using LaneMmaShape = toptlass::gemm::GemmShape<
        LaneM,
        LaneN,
        1>;
    using Policy = toptlass::gemm::warp::MmaSimtPolicy<
      toptlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      toptlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
    >;

     using MmaWarpSimt = toptlass::gemm::warp::MmaSimt<
      WarpShape,      /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,       /// Data type of A elements
      SmemLayoutA,    /// Layout of A matrix (concept: MatrixLayout)
      ElementB,       /// Data type of B elements
      SmemLayoutB,    /// Layout of B matrix (concept: MatrixLayout)
      ElementC,       /// Element type of C matrix
      LayoutC,        /// Layout of C matrix (concept: MatrixLayout)
      Policy          /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    >;

    /// Policy used to define MmaPipelined 
    using MmaPolicy = MmaPolicy<
        MmaWarpSimt,
        MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
        MatrixShape<0, kPaddingN>,    // skew for B matrix to avoid SMEM bank conflicts
        WarpCount::kK
    >;

};


}
}
}