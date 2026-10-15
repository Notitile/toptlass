#pragma once

#include "toptlass/layout/pitch_linear.h"
#include "toptlass/toptlass.h"

namespace toptlass {
namespace transoform {

template <
    typename Shape_,
    int Threads,
    int ElementsPerAccess = 1
>
struct PitchLinearStripminedThreadMap {

    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    /// Number of threads total
    static int const kThreads = Threads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ElementsPerAccess;

    /// Shape of access by each thread
    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    struct Detail {
        static_assert(!(Shape::kContiguous % kElementsPerAccess), "");

        using ShapeVec = layout::PitchLinearShape<
            Shape::kContiguous / kElementsPerAccess,
            Shape::kStrided
        >;

    };

    using Iterations = std::conditional<
        Threads >= Detail::ShapeVec::kContiguous,
        layout::PitchLinearShape<
          1,
          // Redo the comparison here to work around divide by zero compiler
          // error.  The compiler evaluates both path of platform::conditional.
          (Threads >= Detail::ShapeVec::kContiguous
               ? (Detail::ShapeVec::kStrided + (kThreads / Detail::ShapeVec::kContiguous - 1)) /
                     (kThreads / Detail::ShapeVec::kContiguous)
               : 0)>,
      layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                               Detail::ShapeVec::kStrided>>::type;

    using Delta = std::conditional<
        Threads >= Detail::ShapeVec::kContiguous,
        layout::PitchLinearShape<
        1,
        kThreads / Detail::ShapeVec::kContiguous
        >,
        layout::PitchLinearShape<
        kThreads * kElementsPerAccess,
        1
        >
    >::type;

    using StorageShape = std::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<Shape::kContiguous,
                               Iterations::kStrided*(kThreads / Detail::ShapeVec::kContiguous)>,
      layout::PitchLinearShape<Shape::kContiguous, Shape::kStrided>>::type;

    TOPTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {

        return TensorCoord(
            (thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess, 
            thread_id / Detail::ShapeVec::kContiguous);
    }


};

template <typename ThreadMap_>
struct TransposePitchLinearThreadMapSimt {
    /// Underlying ThreadMap
    using ThreadMap = ThreadMap_;

    /// Tensor coordinate
    using TensorCoord = typename ThreadMap::TensorCoord;

    /// Tile shape
    using Shape = typename ThreadMap::Shape;

    /// Number of threads total
    static int const kThreads = ThreadMap::kThreads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static_assert(kElementsPerAccess == 1 , "Simt transpose requires elements per access to be 1");
    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations =
        layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
        ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    static_assert(Iterations::kStrided == 1,
      "Strided iteration has to be one to reuse the same shared store function with those that don't need transpose");

    /// Shape of access by each thread
    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
    using Delta =
        layout::PitchLinearShape<ThreadMap::Delta::kStrided,
        ThreadMap::Delta::kContiguous>;


    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space Note this is slightly different from the one of
    /// PitchLinearWarpRakedThreadMap.
    CUTLASS_HOST_DEVICE
        static TensorCoord initial_offset(int thread_id) {

        TensorCoord coord = ThreadMap::initial_offset(thread_id);

        return TensorCoord(
            coord.strided(),
            coord.contiguous()
        );
    }
};


}
}