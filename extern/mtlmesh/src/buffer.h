// GPU buffer class backed by PyTorch MPS tensors
#pragma once

#include <torch/extension.h>
#include <vector>

namespace mtlmesh {

// Template mapping from C++ types to torch scalar types
template<typename T> struct TorchType;
template<> struct TorchType<float>    { static constexpr auto dtype = torch::kFloat32; static constexpr int ch = 1; };
template<> struct TorchType<int>      { static constexpr auto dtype = torch::kInt32;   static constexpr int ch = 1; };
template<> struct TorchType<int64_t>  { static constexpr auto dtype = torch::kInt64;   static constexpr int ch = 1; };
template<> struct TorchType<uint8_t>  { static constexpr auto dtype = torch::kUInt8;   static constexpr int ch = 1; };

// GPU buffer backed by MPS tensors
// We use torch tensors on MPS device for all GPU memory.
// PyTorch MPS ops handle sort/scan/reduce; underlying MTLBuffers are
// accessed directly for custom Metal kernels.
template<typename T>
struct Buffer {
    torch::Tensor storage;  // MPS tensor
    int64_t size_ = 0;

    Buffer() {}

    bool is_empty() const { return size_ == 0; }
    int64_t size() const { return size_; }

    void resize(int64_t n) {
        if (n <= 0) { size_ = 0; return; }
        if (storage.defined() && storage.numel() >= n) {
            size_ = n;
            return;
        }
        auto opts = torch::dtype(TorchType<T>::dtype).device(torch::kMPS);
        storage = torch::empty({n}, opts);
        size_ = n;
    }

    void free() {
        storage = torch::Tensor();
        size_ = 0;
    }

    void zero() {
        if (size_ > 0 && storage.defined()) {
            storage.slice(0, 0, size_).zero_();
        }
    }

    void fill(T val) {
        if (size_ > 0 && storage.defined()) {
            storage.slice(0, 0, size_).fill_(val);
        }
    }

    // Get a view of the valid portion
    torch::Tensor tensor() const {
        if (size_ <= 0 || !storage.defined()) return torch::Tensor();
        return storage.slice(0, 0, size_);
    }

    // Extend: append space for more elements
    void extend(int64_t n) {
        int64_t new_size = size_ + n;
        if (storage.defined() && storage.numel() >= new_size) {
            size_ = new_size;
            return;
        }
        auto opts = torch::dtype(TorchType<T>::dtype).device(torch::kMPS);
        auto new_storage = torch::empty({new_size}, opts);
        if (size_ > 0 && storage.defined()) {
            new_storage.slice(0, 0, size_).copy_(storage.slice(0, 0, size_));
        }
        storage = new_storage;
        size_ = new_size;
    }
};

} // namespace mtlmesh
