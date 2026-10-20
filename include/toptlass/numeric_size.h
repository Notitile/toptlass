#pragma once

namespace toptlass {

template <typename T>
struct sizeof_bits {
static constexpr int value = int(sizeof(T)*8);
};

}