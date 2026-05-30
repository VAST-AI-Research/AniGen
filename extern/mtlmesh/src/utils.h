#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUDA_CHECK(call)                                \
do {                                                    \
    const cudaError_t error_code = call;                \
    if (error_code != cudaSuccess) {                    \
        TORCH_CHECK(false,                              \
            "[CuMesh] CUDA error:\n",                   \
            "    File:       ", __FILE__, "\n",         \
            "    Line:       ", __LINE__, "\n",         \
            "    Error code: ", error_code, "\n",       \
            "    Error text: ",                         \
            cudaGetErrorString(error_code), "\n");      \
    }                                                   \
} while (0)

namespace cumesh {


/**
 * A GPU buffer class that manages device memory.
 */
template<typename T>
struct Buffer {
    T* ptr;
    size_t size;
    size_t capacity;

    Buffer() : ptr(nullptr), size(0), capacity(0) {}

    bool is_empty() const {
        return size == 0;
    }

    void init(size_t capacity) {
        this->capacity = capacity;
        CUDA_CHECK(cudaMalloc(&ptr, capacity * sizeof(T)));
    }

    void free() {
        if (ptr != nullptr) CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
        size = 0;
        capacity = 0;
    }

    void resize(size_t size) {
        if (size > capacity) {
            free();
            init(size);
        }
        this->size = size;
    }

    void extend(size_t size) {
        size_t new_size = size + this->size;
        if (new_size > capacity) {
            T* new_ptr;
            CUDA_CHECK(cudaMalloc(&new_ptr, new_size * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(new_ptr, ptr, this->size * sizeof(T), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(ptr));
            ptr = new_ptr;
            this->capacity = new_size;
        }
        this->size = new_size;
    }

    void zero() {
        CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(T)));
    }

    void fill(T val) {
        std::vector<T> tmp(size, val);
        CUDA_CHECK(cudaMemcpy(ptr, tmp.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
};


/**
 * Swap the contents of two buffers.
 */
template<typename T1, typename T2>
void swap_buffers(Buffer<T1>& b1, Buffer<T2>& b2) {
    void* b1_ptr = reinterpret_cast<void*>(b1.ptr);
    void* b2_ptr = reinterpret_cast<void*>(b2.ptr);
    size_t b1_capacity_bytes = b1.capacity * sizeof(T1);
    size_t b2_capacity_bytes = b2.capacity * sizeof(T2);
    size_t b1_size_bytes = b1.size * sizeof(T1);
    size_t b2_size_bytes = b2.size * sizeof(T2);
    
    b1.ptr = reinterpret_cast<T1*>(b2_ptr);
    b2.ptr = reinterpret_cast<T2*>(b1_ptr);
    b1.capacity = b2_capacity_bytes / sizeof(T1);
    b2.capacity = b1_capacity_bytes / sizeof(T2);
    b1.size = b2_size_bytes / sizeof(T1);
    b2.size = b1_size_bytes / sizeof(T2);
}


} // namespace cumesh
