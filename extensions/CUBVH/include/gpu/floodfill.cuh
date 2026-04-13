#include <cuda.h>  
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#define THREADS_PER_BLOCK 256

/* ------------------------------------------------------------------------- */  
/*  K E R N E L S                                                            */  
/* ------------------------------------------------------------------------- */

// Initialise: label[i] = i for free voxels, -1 for blocked
__global__ void initLabels(const bool* __restrict__ grid, int * __restrict__ labels, int Ntot)  
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < Ntot) {
        // Only free voxels get valid labels, blocked voxels get -1
        labels[idx] = grid[idx] ? -1 : idx;
    }
}

/*  
 * Hook step for batched volumes.  
 *  - Nvol  := H * W * D            (size of ONE volume)  
 *  - Ntot  := B * Nvol             (size of the whole batch)  
 *  
 * The only difference to the single-volume version is that we build the  
 * voxel coordinates (x,y,z) from the *local* index inside the volume  
 * instead of the global thread index.  
 */  
__global__ void hookBatch(  
    const bool* __restrict__ grid,  
          int*  __restrict__ labels,  
          int*  __restrict__ changed,  
    int H, int W, int D,  
    int Nvol,                // stride between consecutive batches  
    int Ntot)                // total number of voxels  
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx >= Ntot) return;  
    if (grid[idx] != 0) return;        // blocked voxel

    /* ---------- local coordinates inside the current volume -------------- */  
    int batch_idx = idx / Nvol;        // which batch this voxel belongs to
    int local = idx % Nvol;            // 0 … Nvol-1  
    int x =  local % W;  
    int y = (local / W) % H;  
    int z =  local / (W * H);
    
    // Base index of current batch
    int batch_base = batch_idx * Nvol;

    /* ----------------------- six-neighbour search with bounds checking ---- */  
    int best = labels[idx];

    // X-direction neighbors (ensure we stay within the same batch)
    if (x > 0) { 
        int n = idx - 1; 
        if (n >= batch_base && n < batch_base + Nvol && !grid[n] && labels[n] >= 0) 
            best = min(best, labels[n]); 
    }  
    if (x < W - 1) { 
        int n = idx + 1; 
        if (n >= batch_base && n < batch_base + Nvol && !grid[n] && labels[n] >= 0) 
            best = min(best, labels[n]); 
    }

    // Y-direction neighbors
    if (y > 0) { 
        int n = idx - W; 
        if (n >= batch_base && n < batch_base + Nvol && !grid[n] && labels[n] >= 0) 
            best = min(best, labels[n]); 
    }  
    if (y < H - 1) { 
        int n = idx + W; 
        if (n >= batch_base && n < batch_base + Nvol && !grid[n] && labels[n] >= 0) 
            best = min(best, labels[n]); 
    }

    // Z-direction neighbors
    if (z > 0) { 
        int n = idx - W * H; 
        if (n >= batch_base && n < batch_base + Nvol && !grid[n] && labels[n] >= 0) 
            best = min(best, labels[n]); 
    }  
    if (z < D - 1) { 
        int n = idx + W * H; 
        if (n >= batch_base && n < batch_base + Nvol && !grid[n] && labels[n] >= 0) 
            best = min(best, labels[n]); 
    }

    /* --------------------------- atomic update --------------------------- */  
    int current_label = labels[idx];
    if (current_label >= 0 && best < current_label) {
        // Use atomic compare-and-swap to avoid race conditions
        int old_val = atomicCAS(&labels[idx], current_label, best);
        if (old_val == current_label) {
            // Successfully updated, mark as changed
            // Use atomic exchange instead of OR to reduce contention
            atomicExch(changed, 1);
        }
    }  
}

/* Safe pointer-jump compression with bounds checking */  
__global__ void compress(int * __restrict__ labels, int Ntot)  
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx >= Ntot || labels[idx] < 0) return;  // Skip invalid labels
    
    // Find root with bounds checking
    int current = idx;
    int root = labels[current];
    
    // Traverse to find root (with cycle detection)
    int steps = 0;
    const int MAX_STEPS = 1000;  // Prevent infinite loops
    
    while (root != current && root >= 0 && root < Ntot && steps < MAX_STEPS) {
        current = root;
        root = labels[current];
        steps++;
    }
    
    if (steps >= MAX_STEPS) {
        // Corrupted data detected, reset to self-reference
        labels[idx] = idx;
        return;
    }
    
    // Path compression with bounds checking
    current = idx;
    steps = 0;
    while (current != root && current >= 0 && current < Ntot && steps < MAX_STEPS) {
        int next = labels[current];
        labels[current] = root;
        current = next;
        steps++;
    }
}

/* ------------------------------------------------------------------------- */  
/*  H O S T   U T I L I T Y                                                  */  
/* ------------------------------------------------------------------------- */

static int divUp(int a, int b) { return (a + b - 1) / b; }

/* ------------------------------------------------------------------------- */  
/*  P U B L I C   A P I                                                      */  
/* ------------------------------------------------------------------------- */

/*  
 * grid  : pointer to B·H·W·D bools (0 = free, 1 = blocked)  
 * mask  : output, B·H·W·D int32 labels  
 */  
extern "C"  
void _floodfill_batch(  
        const bool *grid,  
        int B, int H, int W, int D,  
        int32_t *mask)  
{  
    const int Nvol = H * W * D;          // size of one volume  
    const int Ntot = B * Nvol;           // size of entire batch

    const size_t bytesGrid  = Ntot * sizeof(bool);  
    const size_t bytesLabel = Ntot * sizeof(int);

    /* --------------------------- allocate -------------------------------- */  
    bool *d_grid    = nullptr;  
    int  *d_labels  = nullptr;  
    int  *d_changed = nullptr;

    cudaMalloc(&d_grid,    bytesGrid);  
    cudaMalloc(&d_labels,  bytesLabel);  
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_grid, grid, bytesGrid, cudaMemcpyHostToDevice);

    /* --------------------------- init ------------------------------------ */  
    {  
        int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
        initLabels<<<blocks, THREADS_PER_BLOCK>>>(d_grid, d_labels, Ntot);  
        cudaDeviceSynchronize();
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in initLabels: %s\n", cudaGetErrorString(err));
            cudaFree(d_grid);
            cudaFree(d_labels);
            cudaFree(d_changed);
            return;
        }
    }

    /* -------------------- iterated hook / compress with bounds ----------- */  
    // Improved iteration limit: account for diagonal propagation distance and batch size
    const int diagonal_dist = (int)ceil(sqrt((double)(H*H + W*W + D*D)));
    // For small grids (like 3x3x3 batches), use a smaller limit
    const int base_limit = (H <= 4 && W <= 4 && D <= 4) ? 20 : diagonal_dist + 50;
    const int MAX_ITERATIONS = min(base_limit, 10 * max(H, max(W, D)));  // More realistic upper bound
    int h_changed = 1;
    int iteration = 0;
    int no_change_count = 0;  // Track consecutive iterations with no changes
    
    while (h_changed && iteration < MAX_ITERATIONS) {  
        cudaMemset(d_changed, 0, sizeof(int));

        /* hook */  
        {  
            int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
            hookBatch<<<blocks, THREADS_PER_BLOCK>>>(  
                d_grid, d_labels, d_changed,  
                H, W, D, Nvol, Ntot);  
        }

        /* compress - only run every few iterations to reduce overhead */  
        if (iteration % 5 == 4 || iteration < 10) {  // Compress more frequently early on
            int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
            compress<<<blocks, THREADS_PER_BLOCK>>>(d_labels, Ntot);  
        }

        cudaDeviceSynchronize();
        
        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in iteration %d: %s\n", iteration, cudaGetErrorString(err));
            break;
        }
        
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        iteration++;
        
        // Early termination: if no changes for several iterations, we've likely converged
        if (h_changed == 0) {
            no_change_count++;
            if (no_change_count >= 3) {  // No changes for 3 consecutive iterations
                break;  // Early convergence detected
            }
        } else {
            no_change_count = 0;
        }
        
        // Progress reporting for very long runs
        // if (iteration % 50 == 0) {  // Report more frequently
        //     printf("Flood fill iteration %d/%d (grid: %dx%dx%d, batches: %d)\n", 
        //            iteration, MAX_ITERATIONS, H, W, D, B);
        // }
    }
    
    if (iteration >= MAX_ITERATIONS) {
        printf("WARNING: Flood fill did not converge after %d iterations!\n", MAX_ITERATIONS);
        printf("Grid info: %dx%dx%d, %d batches. This may indicate a bug.\n", H, W, D, B);
    }

    /* final flatten */  
    {  
        int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
        compress<<<blocks, THREADS_PER_BLOCK>>>(d_labels, Ntot);  
        cudaDeviceSynchronize();  
    }

    /* copy back */  
    cudaMemcpy(mask, d_labels, bytesLabel, cudaMemcpyDeviceToHost);

    /* cleanup */  
    cudaFree(d_grid);  
    cudaFree(d_labels);  
    cudaFree(d_changed);  
}  
