#pragma once

namespace toptlass{
namespace gemm{
namespace warp{

template <typename OperatorClass>
struct WarpSize {
    static int const value = 32;
};

}
}
}