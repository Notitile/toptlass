#pragma once

#include "toptlass/arch/mma.h"

namespace toptlass {
namespace gemm {
namespace device {


template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultGemmConfiguration;

template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementAccumulator>
struct DefaultGemmConfiguration<
    arch::OpClassSimt,
    ArchTag,
    ElementA,
    ElementB,
    ElementC,
    ElementAccumulator>{
    
    static int const kAlignmentA = 1;
    static int const kAlignmentB = 1;
    using ThreadblockShape = GemmShape<64,64,8>;
    using WarpShape = GemmShape<2,4,8>;
    using InstructionShape = GemmShape<1,1,1>;
    static int const kStages = 2;
     
    using Operator = arch::OpMultiplyAdd;

};

}
}
}