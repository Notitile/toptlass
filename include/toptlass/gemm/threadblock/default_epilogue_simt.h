/*! \file
  \brief Epilogue for threadblock scoped GEMMs using SIMT.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/numeric_types.h"
#include "toptlass/array.h"

#include "toptlass/arch/mma.h"

#include "toptlass/gemm/gemm.h"
#include "toptlass/gemm/warp/mma.h"

#include "toptlass/epilogue/thread/linear_combination.h"
#include "toptlass/epilogue/thread/linear_combination_clamp.h"
#include "toptlass/epilogue/thread/linear_combination_relu.h"
#include "toptlass/epilogue/thread/linear_combination_gelu.h"
#include "toptlass/epilogue/thread/linear_combination_sigmoid.h"
#include "toptlass/epilogue/thread/linear_combination_planar_complex.h"
#include "toptlass/epilogue/thread/conversion_op.h"
#include "toptlass/epilogue/thread/reduction_op.h"

#include "toptlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "toptlass/epilogue/warp/fragment_iterator_simt.h"
#include "toptlass/epilogue/warp/tile_iterator_simt.h"
#include "toptlass/epilogue/threadblock/default_thread_map_simt.h"
#include "toptlass/transform/pitch_linear_thread_map.h"

#include "toptlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "toptlass/epilogue/threadblock/predicated_tile_iterator_conv.h"
#include "toptlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "toptlass/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "toptlass/epilogue/threadblock/predicated_tile_iterator_direct_conv.h" 
#include "toptlass/epilogue/threadblock/shared_load_iterator.h"
#include "toptlass/epilogue/threadblock/shared_load_iterator_pitch_linear.h"
#include "toptlass/epilogue/threadblock/epilogue.h"
#include "toptlass/epilogue/threadblock/epilogue_depthwise.h"

#include "toptlass/layout/permute.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace toptlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute,
  conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
  int Rank = 4
>
struct DefaultEpilogueSimt {

  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;
  static conv::StrideSupport const kStrideSupport = StrideSupport;
  static int const kRank = Rank;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename toptlass::epilogue::threadblock::DefaultThreadMapSimt<
    Shape,
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::Policy,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  static bool const UseCUDAStore = std::is_same<ElementOutput, double>::value;

  using PackedOutputTileIterator = toptlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore
  >;

  using StridedOutputTileIterator = toptlass::epilogue::threadblock::PredicatedTileIteratorConv<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore,
    kRank
  >;

  using OutputTileIterator = typename std::conditional<StrideSupport == toptlass::conv::StrideSupport::kUnity,
                                                            PackedOutputTileIterator,
                                                            StridedOutputTileIterator>::type;

  using AccumulatorFragmentIterator = toptlass::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using WarpTileIterator = toptlass::epilogue::warp::TileIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = toptlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = toptlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaSimt,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueSimtStridedDgrad {

  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename toptlass::epilogue::threadblock::DefaultThreadMapSimt<
    Shape,
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::Policy,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = toptlass::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
    OutputTileThreadMap,
    ElementOutput
  >;

  using AccumulatorFragmentIterator = toptlass::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using WarpTileIterator = toptlass::epilogue::warp::TileIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = toptlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = toptlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaSimt,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  int Rank,
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueSimtAffineRankN {

  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename toptlass::epilogue::threadblock::DefaultThreadMapSimt<
    Shape,
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::Policy,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = toptlass::epilogue::threadblock::PredicatedTileIteratorAffineRankN<
    OutputTileThreadMap,
    ElementOutput,
    Rank
  >;

  using AccumulatorFragmentIterator = toptlass::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using WarpTileIterator = toptlass::epilogue::warp::TileIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = toptlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = toptlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaSimt,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <typename Shape_,        // ThreadBlock Shape
          typename WarpMmaSimt_,  // mma_depthwise_simt
          typename OutputOp_,
          int ElementsPerAccess_,
          typename ThreadOutputShape_ = toptlass::conv::TensorNHWCShape<1, 1, 1, 1>,
          typename ThreadBlockOutputShape_ = toptlass::conv::TensorNHWCShape<1, 1, 1, 1> >
struct DefaultDirectConvEpilogueSimt {
  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using WarpShape = typename WarpMmaSimt::Shape;
  using OutputOp = OutputOp_;
  using ThreadOutputShape = ThreadOutputShape_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  static int const kElementsPerAccess = ElementsPerAccess_;


  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;

  /// Number of threads total
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN
  >;

  static int const kWarpSize = toptlass::gemm::warp::WarpSize<arch::OpClassSimt>::value;

  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Thread map
  //
  
  using OutputTileThreadMap = toptlass::transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<ThreadBlockOutputShape::kC, ThreadBlockOutputShape::kNHW>,
    kThreads,
    kElementsPerAccess
  >;


  using OutputTileIterator = toptlass::epilogue::threadblock::PredicatedTileIteratorDirectConv<
    OutputTileThreadMap,
    ElementOutput,
    ThreadOutputShape,
    ThreadBlockOutputShape 
  >;

  using AccumulatorFragmentIterator = toptlass::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;
  
  using WarpTileIterator = toptlass::epilogue::warp::TileIteratorSimtDirect2dConv<
    typename WarpMmaSimt::Shape,
    ThreadOutputShape,
    ThreadBlockOutputShape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = toptlass::epilogue::threadblock::SharedLoadIteratorPitchLinear<
    OutputTileThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;
  //
  // Define the epilogue
  //
  using Epilogue = toptlass::epilogue::threadblock::EpilogueDepthwise<
    Shape,
    ThreadOutputShape,
    ThreadBlockOutputShape,
    WarpMmaSimt,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace toptlass

/////////////////////////////////////////////////////////////////////////////////////////////////
