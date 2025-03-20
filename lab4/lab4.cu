#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call)       \
do {                    \
    cudaError_t status = call;          \
    if  (status != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));   \
        exit(0);                        \
    }                                   \
} while (0)

struct comparator {
  __host__ __device__ bool operator()(double a, double b) {
    return fabs(a) < fabs(b);
  }
};

int findMaxRow(double* d_A_ptr, int n, int i) {
  thrust::device_ptr<double> A_i = thrust::device_pointer_cast(d_A_ptr + i * n);
  return (thrust::max_element(A_i + i, A_i + n, comparator()) - A_i);
}

__global__ void swapA(double* A, int n, int i, int maxRow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = blockDim.x * gridDim.x;
  for (int j = idx; j < n; j += offsetx) {
    double temp = A[j * n + i];
    A[j * n + i] = A[j * n + maxRow];
    A[j * n + maxRow] = temp;
  }
}

__global__ void swapB(double* b, int n, int i, int maxRow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    double temp_b = b[i];
    b[i] = b[maxRow];
    b[maxRow] = temp_b;
  }
}

__global__ void kernelA(double* A, int n, int i) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;

  for (int r = idx + i + 1; r < n; r += offsetx) {
    if (r > i) {
      double temp = A[i * n + r] / A[i * n + i];
      for (int c = idy + i + 1; c < n; c += offsety) {
        A[c * n + r] -= temp * A[c * n + i];
      }
    }
  }
}

__global__ void kernelB(double* A, double* b, int n, int i) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;

  for (int r = idx + i + 1; r < n; r += offsetx) {
    if (r > i) {
      double temp = A[i * n + r] / A[i * n + i];
      if (idy == 0) b[r] -= temp * b[i];
    }
  }
}

int main() {
  int n;
  scanf("%d", &n);

  double* A = (double*)malloc(n * n * sizeof(double));
  double* b = (double*)malloc(n * sizeof(double));
  double* res = (double*)malloc(n * sizeof(double));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      scanf("%lf", &A[j * n + i]);
    }
  }

  for (int i = 0; i < n; i++) {
    scanf("%lf", &b[i]);
  }

  double* d_A_ptr, * d_b;

  CSC(cudaMalloc((void**)&d_A_ptr, n * n * sizeof(double)));
  CSC(cudaMalloc((void**)&d_b, n * sizeof(double)));

  CSC(cudaMemcpy(d_A_ptr, A, n * n * sizeof(double), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

  for (int i = 0; i < n; i++) {
    int maxRow = findMaxRow(d_A_ptr, n, i);
    if (maxRow != i) {
      swapA << <512, 512 >> > (d_A_ptr, n, i, maxRow);
      CSC(cudaDeviceSynchronize());
      swapB << <512, 512 >> > (d_b, n, i, maxRow);
    }
    CSC(cudaDeviceSynchronize());
    kernelA << <dim3(32, 32), dim3(32, 32) >> > (d_A_ptr, n, i);
    CSC(cudaDeviceSynchronize());
    kernelB << <dim3(32, 32), dim3(32, 32) >> > (d_A_ptr, d_b, n, i);
    CSC(cudaDeviceSynchronize());
  }
  CSC(cudaDeviceSynchronize());
  CSC(cudaGetLastError());

  CSC(cudaMemcpy(A, d_A_ptr, n * n * sizeof(double), cudaMemcpyDeviceToHost));
  CSC(cudaMemcpy(b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i = n - 1; i >= 0; i--) {
    res[i] = b[i] / A[i * n + i];
    for (int k = 0; k < i; k++) {
      b[k] -= A[i * n + k] * res[i];
    }
  }

  for (int i = 0; i < n; ++i) {
    printf("%.10e ", res[i]);
  }

  free(A);
  free(b);
  free(res);
  cudaFree(d_A_ptr);
  cudaFree(d_b);

  return 0;
}
