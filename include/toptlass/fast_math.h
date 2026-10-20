#pragma once

#include "toptlass/toptlass.h"
#include <cstdint>
#include <cmath>
#include <type_traits>


namespace toptlass{

template <typename value_t>
TOPTLASS_HOST_DEVICE
TOPTLASS_CONSTEXPR_IF_CXX17
value_t lcm_cxx11(value_t a, value_t b) {
    return value_t{};
}




}