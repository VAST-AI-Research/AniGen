#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
// Added for memory-optimized pipeline
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

// Alias for 3D point (float3 is a built-in CUDA vector type)
using V3f = float3;
// Triangle index structure
struct Tri { int v0, v1, v2; };

// Lookup tables (partial placeholder). 
__device__ __constant__ int edgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   
};

__device__ __constant__ int triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// Device helper: corner offset vectors for a voxel (assuming voxel coordinate = corner 0 position).
__device__ __constant__ int3 cornerOffset[8] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

// Edge i  connects EDGE_CORNERS[i][0]  ↔  EDGE_CORNERS[i][1]
// Matches the edge order required by edgeTable & triTable
__device__ __constant__ int EDGE_CORNERS[12][2] = {
    {0,1},{1,2},{2,3},{3,0},   // 0–3  bottom square, X Y X Y
    {4,5},{5,6},{6,7},{7,4},   // 4–7  top    square
    {0,4},{1,5},{2,6},{3,7}    // 8–11 vertical edges (Z)
};


// Device struct to represent an edge-vertex key (to deduplicate vertices)
struct EdgeKey {
    int x, y, z;
    unsigned char axis;
    __host__ __device__ bool operator==(const EdgeKey& o) const {
        return x == o.x && y == o.y && z == o.z && axis == o.axis;
    }
    __host__ __device__ bool operator<(const EdgeKey& o) const {
        if (x < o.x) return true; if (x > o.x) return false;
        if (y < o.y) return true; if (y > o.y) return false;
        if (z < o.z) return true; if (z > o.z) return false;
        return axis < o.axis;
    }
};

// Functors to avoid device-only lambdas with Thrust templates
struct HeadFlagFunctor {
    const EdgeKey* keys;
    __host__ __device__ int operator()(int i) const {
        if (i == 0) return 1;
        EdgeKey a = keys[i];
        EdgeKey b = keys[i - 1];
        return (a == b) ? 0 : 1;
    }
};

struct IsNonZero {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

struct MinusOne {
    __host__ __device__ int operator()(int x) const { return x - 1; }
};

struct RemapTri {
    const int* map;
    __host__ __device__ void operator()(Tri& tri) const {
        tri.v0 = map[tri.v0];
        tri.v1 = map[tri.v1];
        tri.v2 = map[tri.v2];
    }
};

// CUDA kernel: classify voxels, count intersections and triangles
__global__ void classifyVoxels(const int* coords, const float* corners,
                               int N, float iso,
                               int* vertCount, int* triCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Load the 8 corner values for voxel i
    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }
    // Determine cube index (bitmask of corners < iso)
    int cubeIndex = 0;
    if (v[0] < iso) cubeIndex |= 1;
    if (v[1] < iso) cubeIndex |= 2;
    if (v[2] < iso) cubeIndex |= 4;
    if (v[3] < iso) cubeIndex |= 8;
    if (v[4] < iso) cubeIndex |= 16;
    if (v[5] < iso) cubeIndex |= 32;
    if (v[6] < iso) cubeIndex |= 64;
    if (v[7] < iso) cubeIndex |= 128;
    // Lookup intersected edges
    int mask = edgeTable[cubeIndex];
    // Count set bits in mask (number of intersection vertices)
    // __popc is efficient built-in for counting bits in an int.
    int numVerts = __popc(mask & 0xFFF);  // mask is 12-bit at most
    vertCount[i] = numVerts;
    // Count triangles for this cube configuration using triTable
    int numTris = 0;
    // triTable[cubeIndex][...] contains groups of 3 edge indices for each triangle, terminated by -1
    for (int t = 0; t < 15; t += 3) {
        if (triTable[cubeIndex][t] == -1) break;
        numTris++;
    }
    triCount[i] = numTris;
}

__global__ void generateVertices(const int* coords, const float* corners,
                                 const int* prefixVert, int N, float iso,
                                 EdgeKey* outKeys, V3f* outVerts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int base = prefixVert[i];             // first vertex slot for this voxel
    int vx   = coords[i*3+0];
    int vy   = coords[i*3+1];
    int vz   = coords[i*3+2];

    float v[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) v[c] = corners[i*8 + c];

    // --- cube index & edge mask ---------------------------------------------
    int ci = 0;
    if (v[0] < iso) ci |= 1;   if (v[1] < iso) ci |= 2;
    if (v[2] < iso) ci |= 4;   if (v[3] < iso) ci |= 8;
    if (v[4] < iso) ci |= 16;  if (v[5] < iso) ci |= 32;
    if (v[6] < iso) ci |= 64;  if (v[7] < iso) ci |= 128;
    int mask = edgeTable[ci];
    if (mask == 0) return;

    int outOfs = 0;   // how many verts already emitted for this voxel

    // ------------------------------------------------------------------------
    #pragma unroll
    for (int e = 0; e < 12; ++e)
    {
        if (!(mask & (1 << e))) continue;

        int a = EDGE_CORNERS[e][0];
        int b = EDGE_CORNERS[e][1];

        float va = v[a], vb = v[b];
        float denom = vb - va;
        float t;
        if (fabsf(denom) < 1e-30f)
            t = 0.5f;                       // degenerate edge → midpoint
        else {
            t = (iso - va) / denom;         // interpolate
            t = fminf(fmaxf(t, 0.f), 1.f);  // clamp numerical noise
        }

        int3 offA = cornerOffset[a];
        int3 offB = cornerOffset[b];
        float3 pA = make_float3(vx + offA.x, vy + offA.y, vz + offA.z);
        float3 pB = make_float3(vx + offB.x, vy + offB.y, vz + offB.z);
        float3 P  = { pA.x + t*(pB.x-pA.x),
                      pA.y + t*(pB.y-pA.y),
                      pA.z + t*(pB.z-pA.z) };

        // ------------ build dedup key (lower corner + axis) ------------------
        EdgeKey key;
        key.x = (offA.x < offB.x ? vx+offA.x : vx+offB.x);
        key.y = (offA.y < offB.y ? vy+offA.y : vy+offB.y);
        key.z = (offA.z < offB.z ? vz+offA.z : vz+offB.z);
        key.axis = (offA.x != offB.x) ? 0 : (offA.y != offB.y ? 1 : 2);

        outVerts[base + outOfs] = P;
        outKeys [base + outOfs] = key;
        ++outOfs;
    }
    // now outOfs == __popc(mask)  ✔
}

// CUDA kernel: generate triangles (with *old* vertex indices, will remap later)
__global__ void generateTriangles(const int* coords, const float* corners,
                                  const int* prefixVert, const int* prefixTri,
                                  int N, float iso, Tri* outTris) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int triStart = prefixTri[i];    // starting index in triangle array for voxel i
    int baseVert = prefixVert[i];   // starting index of this voxel's vertices in vertex array
    // Load corner values
    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = corners[i*8 + j];
    }
    // Compute cubeIndex and edge mask again
    int cubeIndex = 0;
    if (v[0] < iso) cubeIndex |= 1;
    if (v[1] < iso) cubeIndex |= 2;
    if (v[2] < iso) cubeIndex |= 4;
    if (v[3] < iso) cubeIndex |= 8;
    if (v[4] < iso) cubeIndex |= 16;
    if (v[5] < iso) cubeIndex |= 32;
    if (v[6] < iso) cubeIndex |= 64;
    if (v[7] < iso) cubeIndex |= 128;
    int mask = edgeTable[cubeIndex];
    if (mask == 0) return;
    // Iterate over triangle entries in triTable
    int outIdx = 0;
    for (int t = 0; t < 15; t += 3) {
        int e0 = triTable[cubeIndex][t];
        if (e0 == -1) break;  // end of list
        int e1 = triTable[cubeIndex][t+1];
        int e2 = triTable[cubeIndex][t+2];
        // Compute original (pre-deduplication) vertex indices for each edge
        // Count how many intersection vertices in this voxel come before the one on edge e
        // i.e., popcount of all lower-numbered edges in the mask.
        int off0 = __popc(mask & ((1 << e0) - 1));
        int off1 = __popc(mask & ((1 << e1) - 1));
        int off2 = __popc(mask & ((1 << e2) - 1));
        Tri tri;
        tri.v0 = baseVert + off0;
        tri.v1 = baseVert + off2; // flip to make sure the triangle is counter-clockwise
        tri.v2 = baseVert + off1;
        outTris[triStart + outIdx] = tri;
        outIdx++;
    }
}

// Device kernel to ensure corner consistency using atomic operations
__global__ void collectCornerContributions(const int* coords, const float* corners, int N,
                                          int* corner_x, int* corner_y, int* corner_z, float* corner_vals, int* corner_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * 8) return;
    
    int voxel_id = tid / 8;
    int corner_id = tid % 8;
    
    // Get voxel position
    int vx = coords[voxel_id * 3 + 0];
    int vy = coords[voxel_id * 3 + 1]; 
    int vz = coords[voxel_id * 3 + 2];
    
    // Get corner position
    int3 offset = cornerOffset[corner_id];
    int cx = vx + offset.x;
    int cy = vy + offset.y;
    int cz = vz + offset.z;
    
    // Store corner data
    corner_x[tid] = cx;
    corner_y[tid] = cy;
    corner_z[tid] = cz;
    corner_vals[tid] = corners[voxel_id * 8 + corner_id];
    corner_counts[tid] = 1;
}

__global__ void updateCornerValues(const int* coords, float* corners, int N,
                                 const int* unique_x, const int* unique_y, const int* unique_z, 
                                 const float* avg_vals, int num_unique) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * 8) return;
    
    int voxel_id = tid / 8;
    int corner_id = tid % 8;
    
    // Get voxel position
    int vx = coords[voxel_id * 3 + 0];
    int vy = coords[voxel_id * 3 + 1]; 
    int vz = coords[voxel_id * 3 + 2];
    
    // Get corner position
    int3 offset = cornerOffset[corner_id];
    int cx = vx + offset.x;
    int cy = vy + offset.y;
    int cz = vz + offset.z;
    
    // Binary search for this corner coordinate
    int left = 0, right = num_unique - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        int mx = unique_x[mid];
        int my = unique_y[mid]; 
        int mz = unique_z[mid];
        
        if (cx < mx || (cx == mx && cy < my) || (cx == mx && cy == my && cz < mz)) {
            right = mid - 1;
        } else if (cx > mx || (cx == mx && cy > my) || (cx == mx && cy == my && cz > mz)) {
            left = mid + 1;
        } else {
            // Found exact match, update corner value
            corners[voxel_id * 8 + corner_id] = avg_vals[mid];
            break;
        }
    }
}

// Helper structure for corner data sorting
struct CornerData {
    int x, y, z;
    float value;
    int count;
    
    __host__ __device__ bool operator<(const CornerData& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
    
    __host__ __device__ bool operator==(const CornerData& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Reduction operator for averaging
struct CornerAverage {
    __host__ __device__ CornerData operator()(const CornerData& a, const CornerData& b) const {
        CornerData result;
        result.x = a.x;  // coordinates should be same
        result.y = a.y;
        result.z = a.z;
        result.value = a.value + b.value;  // sum values
        result.count = a.count + b.count;  // sum counts
        return result;
    }
};

// Host function: sparse marching cubes
inline std::pair<thrust::device_vector<V3f>, thrust::device_vector<Tri>>
_sparse_marching_cubes(const int* d_coords, const float* d_corners, int N, float iso, bool ensure_consistency, cudaStream_t stream) {
    // Output containers
    thrust::device_vector<V3f> vertices; 
    thrust::device_vector<Tri> triangles;
    if (N <= 0) {
        // No voxels, return empty mesh
        return {vertices, triangles};
    }

    // Create a copy of corners data if we need to ensure consistency
    thrust::device_vector<float> corners_copy;
    const float* d_corners_to_use = d_corners;
    
    if (ensure_consistency) {
        // Copy original corner data
        corners_copy.resize(N * 8);
        thrust::copy(thrust::cuda::par.on(stream), 
                     d_corners, d_corners + N * 8, 
                     corners_copy.begin());
        
        // Total number of corner instances (8 per voxel)
        const int total_corners = N * 8;
        
        // Create arrays to store corner data
        thrust::device_vector<int> corner_x(total_corners);
        thrust::device_vector<int> corner_y(total_corners);
        thrust::device_vector<int> corner_z(total_corners);
        thrust::device_vector<float> corner_vals(total_corners);
        thrust::device_vector<int> corner_counts(total_corners);
        
        // Collect all corner contributions
        int threads_corner = 256;
        int blocks_corner = (total_corners + threads_corner - 1) / threads_corner;
        collectCornerContributions<<<blocks_corner, threads_corner, 0, stream>>>(
            d_coords, d_corners, N,
            thrust::raw_pointer_cast(corner_x.data()),
            thrust::raw_pointer_cast(corner_y.data()),
            thrust::raw_pointer_cast(corner_z.data()),
            thrust::raw_pointer_cast(corner_vals.data()),
            thrust::raw_pointer_cast(corner_counts.data()));
        
        // Create corner data structure for sorting and averaging
        thrust::device_vector<CornerData> corner_data(total_corners);
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(total_corners),
                          corner_data.begin(),
                          [corner_x_ptr = thrust::raw_pointer_cast(corner_x.data()),
                           corner_y_ptr = thrust::raw_pointer_cast(corner_y.data()),
                           corner_z_ptr = thrust::raw_pointer_cast(corner_z.data()),
                           corner_vals_ptr = thrust::raw_pointer_cast(corner_vals.data()),
                           corner_counts_ptr = thrust::raw_pointer_cast(corner_counts.data())] __device__ (int i) {
                              CornerData data;
                              data.x = corner_x_ptr[i];
                              data.y = corner_y_ptr[i];
                              data.z = corner_z_ptr[i];
                              data.value = corner_vals_ptr[i];
                              data.count = corner_counts_ptr[i];
                              return data;
                          });
        
        // Sort by corner coordinates
        thrust::sort(thrust::cuda::par.on(stream), corner_data.begin(), corner_data.end());
        
        // Reduce by key to get average for each unique corner
        thrust::device_vector<CornerData> unique_corners(total_corners);
        thrust::device_vector<CornerData> corner_sums(total_corners);
        
        auto new_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            corner_data.begin(), corner_data.end(),
            corner_data.begin(),
            unique_corners.begin(),
            corner_sums.begin(),
            [] __device__ (const CornerData& a, const CornerData& b) {
                return a.x == b.x && a.y == b.y && a.z == b.z;
            },
            CornerAverage());
        
        int num_unique = new_end.first - unique_corners.begin();
        unique_corners.resize(num_unique);
        corner_sums.resize(num_unique);
        
        // Compute averages and create coordinate arrays
        thrust::device_vector<int> unique_x(num_unique);
        thrust::device_vector<int> unique_y(num_unique);
        thrust::device_vector<int> unique_z(num_unique);
        thrust::device_vector<float> avg_vals(num_unique);
        
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(num_unique),
                          thrust::make_zip_iterator(thrust::make_tuple(
                              unique_x.begin(), unique_y.begin(), unique_z.begin(), avg_vals.begin())),
                          [unique_corners_ptr = thrust::raw_pointer_cast(unique_corners.data()),
                           corner_sums_ptr = thrust::raw_pointer_cast(corner_sums.data())] __device__ (int i) {
                              const CornerData& corner = unique_corners_ptr[i];
                              const CornerData& sum = corner_sums_ptr[i];
                              float avg = sum.value / sum.count;
                              return thrust::make_tuple(corner.x, corner.y, corner.z, avg);
                          });
        
        // Update corner values using the simpler kernel
        updateCornerValues<<<blocks_corner, threads_corner, 0, stream>>>(
            d_coords,
            thrust::raw_pointer_cast(corners_copy.data()),
            N,
            thrust::raw_pointer_cast(unique_x.data()),
            thrust::raw_pointer_cast(unique_y.data()),
            thrust::raw_pointer_cast(unique_z.data()),
            thrust::raw_pointer_cast(avg_vals.data()),
            num_unique);
        
        d_corners_to_use = thrust::raw_pointer_cast(corners_copy.data());
    }

    // Temporary arrays for counts and prefix sums
    thrust::device_vector<int> vertCount(N);
    thrust::device_vector<int> triCount(N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    classifyVoxels<<<blocks, threads, 0, stream>>>(d_coords, d_corners_to_use, N, iso,
                                                  thrust::raw_pointer_cast(vertCount.data()),
                                                  thrust::raw_pointer_cast(triCount.data()));

    thrust::device_vector<int> prefixVert(N);
    thrust::device_vector<int> prefixTri(N);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           vertCount.begin(), vertCount.end(), prefixVert.begin());
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           triCount.begin(), triCount.end(), prefixTri.begin());

    // Compute totals
    int M = vertCount.empty() ? 0 : (prefixVert.back() + vertCount.back());
    int T = triCount.empty() ? 0 : (prefixTri.back() + triCount.back());

    // Free counts early to lower peak memory
    thrust::device_vector<int>().swap(vertCount);
    thrust::device_vector<int>().swap(triCount);

    // Allocate space for all intersection vertices (M) and their keys
    thrust::device_vector<EdgeKey> keys(M);
    thrust::device_vector<V3f>    verts(M);

    // Generate vertices in parallel
    blocks = (N + threads - 1) / threads;
    generateVertices<<<blocks, threads, 0, stream>>>(d_coords, d_corners_to_use,
                                                    thrust::raw_pointer_cast(prefixVert.data()),
                                                    N, iso,
                                                    thrust::raw_pointer_cast(keys.data()),
                                                    thrust::raw_pointer_cast(verts.data()));

    // Create index array [0, 1, ..., M-1] to track original positions
    thrust::device_vector<int> indices(M);
    if (M > 0) {
        thrust::sequence(thrust::cuda::par.on(stream), indices.begin(), indices.end());

        // Sort by key and reorder verts and indices in one pass (no extra vertsSorted buffer)
        auto zipped_vals = thrust::make_zip_iterator(thrust::make_tuple(verts.begin(), indices.begin()));
        thrust::sort_by_key(thrust::cuda::par.on(stream), keys.begin(), keys.end(), zipped_vals);

        // Build head flags using a transform iterator functor
        EdgeKey* d_keys = thrust::raw_pointer_cast(keys.data());
        auto count_begin = thrust::make_counting_iterator<int>(0);
        auto head_flags = thrust::make_transform_iterator(count_begin, HeadFlagFunctor{d_keys});

        // Compute inclusive scan of head flags directly into group ids (1-based)
        thrust::device_vector<int> mapSortedToUnique(M);
        thrust::inclusive_scan(thrust::cuda::par.on(stream), head_flags, head_flags + M, mapSortedToUnique.begin());

        // Number of unique vertices
        int uniqueCount = 0;
        cudaMemcpyAsync(&uniqueCount, thrust::raw_pointer_cast(mapSortedToUnique.data()) + (M - 1),
                        sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        vertices.resize(uniqueCount);

        // Emit unique vertices by copying only the heads
        thrust::copy_if(thrust::cuda::par.on(stream),
                        verts.begin(), verts.end(),
                        head_flags,
                        vertices.begin(),
                        IsNonZero());

        // Build mapping from original vertex index -> new unique index
        thrust::device_vector<int> mapOrigToNew(M);
        // Convert to 0-based ids in-place and scatter back to original order
        thrust::transform(thrust::cuda::par.on(stream),
                          mapSortedToUnique.begin(), mapSortedToUnique.end(),
                          mapSortedToUnique.begin(),
                          MinusOne());
        thrust::scatter(thrust::cuda::par.on(stream),
                        mapSortedToUnique.begin(), mapSortedToUnique.end(),
                        indices.begin(),
                        mapOrigToNew.begin());

        // Free temporaries no longer needed before allocating triangles
        thrust::device_vector<EdgeKey>().swap(keys);
        thrust::device_vector<V3f>().swap(verts);
        thrust::device_vector<int>().swap(indices);
        thrust::device_vector<int>().swap(mapSortedToUnique);

        // Generate triangles (old indices)
        triangles.resize(T);
        blocks = (N + threads - 1) / threads;
        generateTriangles<<<blocks, threads, 0, stream>>>(d_coords, d_corners_to_use,
                                                         thrust::raw_pointer_cast(prefixVert.data()),
                                                         thrust::raw_pointer_cast(prefixTri.data()),
                                                         N, iso,
                                                         thrust::raw_pointer_cast(triangles.data()));
        // Remap triangle vertex indices to the new deduplicated indices
        if (T > 0) {
            int* d_map = thrust::raw_pointer_cast(mapOrigToNew.data());
            thrust::for_each(thrust::cuda::par.on(stream),
                             triangles.begin(), triangles.end(),
                             RemapTri{d_map});
        }
        // mapOrigToNew freed on scope exit
    } else {
        // No intersections; still need to size triangles correctly
        triangles.resize(T);
        if (T > 0) {
            blocks = (N + threads - 1) / threads;
            generateTriangles<<<blocks, threads, 0, stream>>>(d_coords, d_corners_to_use,
                                                             thrust::raw_pointer_cast(prefixVert.data()),
                                                             thrust::raw_pointer_cast(prefixTri.data()),
                                                             N, iso,
                                                             thrust::raw_pointer_cast(triangles.data()));
            // No remap needed; M==0 implies T should also be 0 for standard MC, but keep safe
        }
    }

    cudaStreamSynchronize(stream);
    return { std::move(vertices), std::move(triangles) };
}
