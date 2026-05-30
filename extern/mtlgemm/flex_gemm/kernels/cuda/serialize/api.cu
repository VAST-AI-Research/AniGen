#include <torch/extension.h>
#include "api.h"
#include "z_order.h"
#include "hilbert.h"


namespace flex_gemm {
namespace serialize {

void z_order_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
) {
    // Call kernel
    if (coords.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::z_order_encode<<<(coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::z_order_encode<<<(coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::z_order_encode(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::z_order_encode(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    }
}


torch::Tensor z_order_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    // Allocate output tensors
    auto coords = torch::empty({codes.size(0), 4}, torch::dtype(torch::kInt32).device(codes.device()));

    // Call kernel
    if (codes.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::z_order_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::z_order_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::z_order_decode(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::z_order_decode(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    }

    return coords;
}


void hilbert_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
) {
    // Call kernel
    if (coords.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::hilbert_encode<<<(coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::hilbert_encode<<<(coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::hilbert_encode(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(codes.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::hilbert_encode(
                coords.size(0),
                reinterpret_cast<const uint32_t*>(coords.data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint64_t*>(codes.data_ptr<int64_t>())
            );
        } else {
            throw std::runtime_error("Unsupported output type");
        }
    }
}


torch::Tensor hilbert_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    // Allocate output tensors
    auto coords = torch::empty({codes.size(0), 4}, torch::dtype(torch::kInt32).device(codes.device()));

    // Call kernel
    if (codes.device().type() == torch::kCUDA) {
        if (codes.dtype() == torch::kInt32) {
            cuda::hilbert_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cuda::hilbert_decode<<<(codes.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    } else {
        if (codes.dtype() == torch::kInt32) {
            cpu::hilbert_decode(
                codes.size(0),
                reinterpret_cast<uint32_t*>(codes.contiguous().data_ptr<int>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else if (codes.dtype() == torch::kInt64) {
            cpu::hilbert_decode(
                codes.size(0),
                reinterpret_cast<uint64_t*>(codes.contiguous().data_ptr<int64_t>()),
                bit_length,
                reinterpret_cast<uint32_t*>(coords.data_ptr<int>())
            );
        } else {
            throw std::runtime_error("Unsupported input type");
        }
    }

    return coords;
}

} // namespace serialize
} // namespace flex_gemm
