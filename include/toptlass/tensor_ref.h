#pragma once


#include "toptlass/subbyte_reference.h"

namespace toptlass {

template <
    typename Element_,
    typename Layout_,
>
class TensorRef {
public:
    using Element = Element_;
    using Layout = Layout_;

    using Reference = typename std::conditional<
        sizeof_bits<Element>::value >= 8,
        Element &,
        SubbyteReference<Element>
        >::type;

    static int const kRank = Layout::kRank;

    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;

    using Stride = typename Layout::Stride;

private:
    Element* ptr_;
    Layout layout_;

public:
    TOPTLASS_HOST_DEVICE
    TensorRef():ptr_(nullptr){

    }

    TOPTLASS_HOST_DEVICE
    TensorRef(
        Element *ptr,
        Layout const &layout
    ):
        ptr_(ptr),layout_(layout){

    }
};

}