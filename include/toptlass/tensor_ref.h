#pragma once


#include "toptlass/subbyte_reference.h"
#include "toptlass/toptlass.h"
#include "toptlass/coord.h"
namespace toptlass {

template <
    typename Element_,
    typename Layout_
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

    using ConstTensorRef = TensorRef<
        typename std::remove_const<Element>::type const,Layout>;

    using NonConstTensorRef = TensorRef<
        typename std::remove_const<Element>::type,Layout>;

    using LongIndex = typename Layout::LongIndex;


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

    template<typename _Magic = int>
    TOPTLASS_DEVICE
    TensorRef(
        NonConstTensorRef const &ref,
        _Magic magic = (typename std::enable_if< ! std::is_same<NonConstTensorRef,TensorRef<Element_,Layout_>>::value,_Magic>::type)0
    ):
      ptr_(ref.data()),layout_(ref.layout()){}

    TOPTLASS_HOST_DEVICE
    ConstTensorRef const_ref() const {
        return ConstTensorRef(ptr_, layout_);
    }

    TOPTLASS_HOST_DEVICE
    NonConstTensorRef non_const_ref() const {
        return NonConstTensorRef(const_cast<typename std::remove_const<Element>::type *>(ptr_), layout_);
    }

    /// Updates only the pointer
    TOPTLASS_HOST_DEVICE
    void reset(Element* ptr = nullptr) {
        ptr_ = ptr;
    }

    /// Updates the pointer and layout object
    TOPTLASS_HOST_DEVICE
    void reset(Element* ptr, Layout const &layout) {
        ptr_ = ptr;
        layout_ = layout;
    }

    /// Returns true if the TensorRef is non-null
    TOPTLASS_HOST_DEVICE
    bool good() const {
        return ptr_ != nullptr;
    }

    /// Returns the pointer to referenced data
    TOPTLASS_HOST_DEVICE
    Element * data() const { return ptr_; }

    /// Returns a reference to the element at a given linear index
    TOPTLASS_HOST_DEVICE
    Reference data(LongIndex idx) const {
        return ReferenceFactory<typename std::remove_const<Element>::type,
                                (sizeof_bits<Element>::value < 8)>::get(ptr_, idx);
    }

    /// Returns the layout object
    TOPTLASS_HOST_DEVICE
    Layout & layout() {
        return layout_;
    }

    /// Returns the layout object
    TOPTLASS_HOST_DEVICE
    Layout layout() const {
        return layout_;
    }

    /// Returns the layout object's stride vector
    TOPTLASS_HOST_DEVICE
    Stride stride() const {
        return layout_.stride();
    }

    /// Returns the layout object's stride vector
    TOPTLASS_HOST_DEVICE
    Stride & stride() {
        return layout_.stride();
    }

    /// Returns the layout object's stride in a given physical dimension
    TOPTLASS_HOST_DEVICE
    typename Layout::Stride::Index stride(int dim) const {
        return layout_.stride().at(dim);
    }

    /// Returns the layout object's stride in a given physical dimension
    TOPTLASS_HOST_DEVICE
    typename Layout::Stride::Index & stride(int dim) {
        return layout_.stride().at(dim);
    }

    /// Computes the offset of an index from the origin of the tensor
    TOPTLASS_HOST_DEVICE
    LongIndex offset(TensorCoord const& coord) const {
        return layout_(coord);
    }

    /// Returns a reference to the element at a given Coord
    TOPTLASS_HOST_DEVICE
    Reference at(TensorCoord const& coord) const {
        return data(offset(coord));
    }

    /// Returns a reference to the element at a given Coord
    TOPTLASS_HOST_DEVICE
    Reference operator[](TensorCoord const& coord) const {
        return data(offset(coord));
    }

    /// Adds an offset to each pointer
    TOPTLASS_HOST_DEVICE
    TensorRef & add_pointer_offset(LongIndex offset_) {
        ptr_ = ReferenceFactory<typename std::remove_const<Element>::type,
            (sizeof_bits<Element>::value < 8)>::add_pointer_offset(ptr_, offset_);
        return *this;
    }

    /// Adds an offset to each pointer
    TOPTLASS_HOST_DEVICE
    TensorRef & add_coord_offset(TensorCoord const &coord) {
        add_pointer_offset(offset(coord));
        return *this;
    }

    /// Returns a TensorRef offset by a given amount
    TOPTLASS_HOST_DEVICE
    TensorRef operator+(TensorCoord const& b) const {
        TensorRef result(*this);
        result.add_coord_offset(b);
        return result;
    }

    /// Returns a TensorRef offset by a given amount
    TOPTLASS_HOST_DEVICE
    TensorRef & operator+=(TensorCoord const& b) {
        add_coord_offset(b);
        return *this;
    }

    /// Returns a TensorRef offset by a given amount
    TOPTLASS_HOST_DEVICE
    TensorRef operator-(TensorCoord const& b) const {
        TensorRef result(*this);
        result.add_pointer_offset(-offset(b));
        return result;
    }

    /// Returns a TensorRef offset by a given amount
    TOPTLASS_HOST_DEVICE
    TensorRef & operator-=(TensorCoord const& b) {
        add_pointer_offset(-offset(b));
        return *this;
    }


};

}