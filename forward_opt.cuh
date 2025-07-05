#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

// Includes:
#include <mxnet/base.h>

// Defines:
#define BLOCK_SIZE 8

namespace mxnet {
namespace op {

// Constant Memory for Single Channel:
__constant__ float singleMask[2500];

__global__ void forward_single_kernel(float *y, const float *x, const int M, const int H, const int W, const int K) {

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Optimization 3: Shared Memory (Tiling)
  // ----------------------------------------------------------------
  extern __shared__ float X_ds[];
  int X_TILE_WIDTH = BLOCK_SIZE + K - 1;

  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define k3d(i2, i1, i0) singleMask[(i2) * (K * K) + (i1) * (K) + i0]
  #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]

  // Indicies:
  int h0 = threadIdx.x;
  int w0 = threadIdx.y;

  int h_base = (blockDim.x * blockIdx.x);
  int w_base = (blockDim.y * blockIdx.y);

  // Optimization 1: Parallelizing Image Processing
  // ----------------------------------------------------------------
  int h_out = h_base + h0;
  int w_out = w_base + w0;
  int b = blockIdx.z;

  // Load Into Shared Memory:
  for (int i = h_out; i < h_base + X_TILE_WIDTH; i += BLOCK_SIZE) {
    for (int j = w_out; j < w_base + X_TILE_WIDTH; j += BLOCK_SIZE)
      if (i < H && j < W) 
        X_ds[(i - h_base) * X_TILE_WIDTH + (j - w_base)] = x3d(b, i, j);
  }

  __syncthreads();

  // Boundary Check:
  if (h_out < H_out && w_out < W_out) {

    // Iterate through each output map:
    for (int m = 0; m < M; m++) {

      float sum = 0.0f;
      
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          sum += X_ds[(h0 + p) * X_TILE_WIDTH + (w0 + q)] * k3d(m, p, q);
        }

      }
      
      // Assign to Output:
      y4d(b, m, h_out, w_out) = sum;

    }
  }

  #undef y4d
  #undef k3d
  #undef x3d
}

__global__ void forward_multi_kernel(float *y, const float *x, const float *k, const int M, const int C, const int H, const int W, const int K) {

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Optimization 3: Shared Memory (Tiling)
  // ----------------------------------------------------------------
  extern __shared__ float shmem[];
  int X_TILE_WIDTH = BLOCK_SIZE + K - 1;

  float* W_ds = &shmem[X_TILE_WIDTH * X_TILE_WIDTH];
  float* X_ds = &shmem[0];

  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Indicies:
  int h0 = threadIdx.x;
  int w0 = threadIdx.y;

  int h_base = (blockDim.x * blockIdx.x);
  int w_base = (blockDim.y * blockIdx.y);

  // Optimization 1: Parallelizing Image Processing
  // ----------------------------------------------------------------
  int h_out = h_base + h0;
  int w_out = w_base + w0;
  int b = blockIdx.z;

  // Iterate through each output map:
  for (int m = 0; m < M; m++) {

    float sum = 0.0f;

    // Iterate through each input map:
    for (int c = 0; c < C; c ++) {
      
      // Load Weight Into Shared Memory:
      if (h0 < K && w0 < K)
        W_ds[h0 * K + w0] = k4d(m, c, h0, w0);

      __syncthreads();

      // Load Input Into Shared Memory:
      for (int i = h_out; i < h_base + X_TILE_WIDTH; i += BLOCK_SIZE) {
        for (int j = w_out; j < w_base + X_TILE_WIDTH; j += BLOCK_SIZE) 
          if (i < H && j < W) 
            X_ds[(i - h_base) * X_TILE_WIDTH + (j - w_base)] = x4d(b, c, i, j);
      }

      __syncthreads();

      if (h_out < H_out && w_out < W_out) {
        for (int p = 0; p < K; p++) 
          for (int q = 0; q < K; q++)
            sum += X_ds[(h0 + p) * X_TILE_WIDTH + (w0 + q)] * W_ds[p * K + q];
      }

      __syncthreads();

    }

    // Assign to Output:
    if (h_out < H_out && w_out < W_out) 
      y4d(b, m, h_out, w_out) = sum;

  }


  #undef y4d
  #undef x4d
  #undef k4d
}

/* 
  This function is called by new-inl.h
  Any code you write should be executed by this function.
  For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {

  // // Use mxnet's CHECK_EQ to do assertions.
  // // Remove this assertion when you do your implementation!
  // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

  const int B = x.shape_[0];
  const int M = y.shape_[1]; // num_filter
  const int C = x.shape_[1];
  const int H = x.shape_[2];
  const int W = x.shape_[3]; // 
  const int K = w.shape_[3]; // Height and Width of Filter


  // Optimization 1: Parallelizing Image Processing
  // ----------------------------------------------------------------
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  dim3 gridDim(ceil(H_out / (float)BLOCK_SIZE), ceil(W_out / (float)BLOCK_SIZE), B);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  // Optimization 2: Different Kernels (Single  vs. Multi Channel)
  // ----------------------------------------------------------------
  if (C == 1) {

    // Constant Memory:
    size_t maskSize = (size_t)((K * K) * M * C);
    cudaMemcpyToSymbol(singleMask, w.dptr_, maskSize * sizeof(float), 0);

    // Optimization 3: Shared Memory (Tiling)
    // ----------------------------------------------------------------
    size_t sharedMem = (BLOCK_SIZE + K - 1) * (BLOCK_SIZE + K - 1);
    sharedMem *= sizeof(float);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_single_kernel<<<gridDim, blockDim, sharedMem>>>(y.dptr_, x.dptr_, M, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

  } else {

    // Optimization 3: Shared Memory (Tiling)
    // ----------------------------------------------------------------
    size_t sharedMem = (BLOCK_SIZE + K - 1) * (BLOCK_SIZE + K - 1) + (K*K);
    sharedMem *= sizeof(float);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_multi_kernel<<<gridDim, blockDim, sharedMem>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

  }

}

/* 
  This tells mxnet how to do an op when it's not a float.
  This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
  assert(0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif