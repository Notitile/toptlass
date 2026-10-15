#pragma once

#include "toptlass/toptlass.h"
#include "toptlass/numeric_size.h"

namespace toptlass {

template <
    typename Element_,
    typename Storage_ = uint8_t,
    class = void>
class SubbyteReference {
public:
    using Element = Element_;
    using Storage = Storage_;
    using StoragePointer = Storage *;

private:
    int const kElementsPerVector = sizeof_bits<Storage>::value /sizeof_bits<Element>::value;

    Storage const kMask = 
        ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ?
            (Storage(1) << sizeof_bits<Element>::value) - Storage(1):
            ~Storage(0));

    StoragePointer ptr_;

    int offset_;

public:
    TOPTLASS_HOST_DEVICE
    SubbyteReference():ptr_(nullptr),offset_(0){}

    TOPTLASS_HOST_DEVICE
    SubbyteReference(
        Element *ptr,
        int64_t offset
    ):
        ptr_(reinterpret_cast<StoragePointer>(ptr)),
        offset_(0){
        
        int64_t offset_in_vectors = offset / kElementsPerVector;
        int64_t offset_in_elements = offset % kElementsPerVector;

        ptr_ += offset_in_vectors;
        offset_ = int(offset_in_elements);

        }
};

}