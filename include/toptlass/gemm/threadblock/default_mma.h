#pragma once

#include "toptlass/gemm/gemm_enumerated_types.h"
#include "toptlass/layout/permute.h"

namespace toptlass {
namespace gemm {
namespace threadblock{

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute
    >
struct DefaultMma;


template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {
    static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

    // Define the MmaCore components
    using MmaCore = typename toptlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
        ElementB, LayoutB, ElementAccumulator, LayoutC,
        arch::OpClassSimt, 2, Operator>;

    // Define iterators over tiles from the A operand
    using IteratorA =
        toptlass::transform::threadblock::PredicatedTileIterator<
            toptlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
            GatherA, PermuteALayout>;

    // Define iterators over tiles from the B operand
    using IteratorB =
        toptlass::transform::threadblock::PredicatedTileIterator<
            toptlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
            GatherB, PermuteBLayout>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = toptlass::gemm::threadblock::MmaPipelined<
        typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
        IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
        LayoutC, typename MmaCore::MmaPolicy>;
                    

};

}
}
}