#pragma once

// Tile sizes for implicit GEMM — can be overridden via -D flags at compile time
#ifndef GEMM_BLOCK_N
#define GEMM_BLOCK_N 64
#endif
#ifndef GEMM_BLOCK_CO
#define GEMM_BLOCK_CO 64
#endif
#ifndef GEMM_BLOCK_K
#define GEMM_BLOCK_K 32
#endif
#ifndef GEMM_THREADS
#define GEMM_THREADS 256
#endif
