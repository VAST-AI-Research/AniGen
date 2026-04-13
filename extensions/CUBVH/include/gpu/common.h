#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <cstdint>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <Eigen/Dense>

namespace cubvh {

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT2 = 1.41421356237309504880f;


// enum class EMeshSdfMode : int {
// 	Watertight,
// 	Raystab,
// 	PathEscape,
// };
// static constexpr const char* MeshSdfModeStr = "Watertight\0Raystab\0PathEscape\0\0";

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
    return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
    if (n_elements <= 0) {
        return;
    }
    kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>((uint32_t)n_elements, args...);
}

inline __host__ __device__ float sign(float x) {
    return copysignf(1.0, x);
}

template <typename T>
__host__ __device__ T clamp(T val, T lower, T upper) {
    return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T>
__host__ __device__ void host_device_swap(T& a, T& b) {
    T c(a); a=b; b=c;
}

inline __host__ __device__ Eigen::Vector3f cylindrical_to_dir(const Eigen::Vector2f& p) {
	const float cos_theta = -2.0f * p.x() + 1.0f;
	const float phi = 2.0f * PI * (p.y() - 0.5f);

	const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
	float sin_phi, cos_phi;
	sincosf(phi, &sin_phi, &cos_phi);

	return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

inline __host__ __device__ float fractf(float x) {
	return x - floorf(x);
}

template <uint32_t N_DIRS>
__device__ __host__ Eigen::Vector3f fibonacci_dir(uint32_t i, const Eigen::Vector2f& offset) {
	// Fibonacci lattice with offset
	float epsilon;
	if (N_DIRS >= 11000) {
		epsilon = 27;
	} else if (N_DIRS >= 890) {
		epsilon = 10;
	} else if (N_DIRS >= 177) {
		epsilon = 3.33;
	} else if (N_DIRS >= 24) {
		epsilon = 1.33;
	} else {
		epsilon = 0.33;
	}

	static constexpr float GOLDEN_RATIO = 1.6180339887498948482045868343656f;
	return cylindrical_to_dir(Eigen::Vector2f{fractf((i+epsilon) / (N_DIRS-1+2*epsilon) + offset.x()), fractf(i / GOLDEN_RATIO + offset.y())});
}

inline __host__ __device__ float safe_divide(float numerator, float denominator, float epsilon = 1e-6f) {
	if (fabs(denominator) < epsilon) {
		if (denominator <= 0)
			return -(numerator / epsilon); 
		else
			return numerator / epsilon; 
	}
	return numerator / denominator;
}

}