#include <stdio.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void compute_histogram(unsigned char* d_input, int* d_histogram, int n) {
  __shared__ int hist[256];
  if (threadIdx.x < 256) {
    hist[threadIdx.x] = 0;
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = gridDim.x * blockDim.x;

  for (int i = idx; i < n; i += offsetx) {
    atomicAdd(&hist[d_input[i]], 1);
  }
  __syncthreads();

  if (threadIdx.x < 256) {
    atomicAdd(&d_histogram[threadIdx.x], hist[threadIdx.x]);
  }
}

__global__ void compute_sum(int* d_histogram, int* d_sum) {
  __shared__ int shared_memory[256];
  shared_memory[threadIdx.x] = d_histogram[threadIdx.x];
  __syncthreads();

  for (int j = 1; j < 256; j *= 2) {
    int i = (threadIdx.x + 1) * j * 2 - 1;
    if (i < 256) {
      shared_memory[i] += shared_memory[i - j];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) shared_memory[256 - 1] = 0;
  __syncthreads();

  for (int j = 256 / 2; j > 0; j /= 2) {
    int i = (threadIdx.x + 1) * j * 2 - 1;
    if (i < 256) {
      int temp = shared_memory[i - j];
      shared_memory[i - j] = shared_memory[i];
      shared_memory[i] += temp;
    }
    __syncthreads();
  }

  d_sum[threadIdx.x] = shared_memory[threadIdx.x];
  __syncthreads();
}

__global__ void kernel_sort(unsigned char* d_input, unsigned char* d_output, int* d_sum, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetx = gridDim.x * blockDim.x;
  for (int i = idx; i < n; i += offsetx) {
    d_output[atomicAdd(&d_sum[d_input[i]], 1)] = d_input[i];
  }
}

int main() {
  int n;
  fread(&n, sizeof(int), 1, stdin);
  unsigned char* input = (unsigned char*)malloc(n * sizeof(unsigned char));
  unsigned char* output = (unsigned char*)malloc(n * sizeof(unsigned char));
  fread(input, sizeof(unsigned char), n, stdin);

  unsigned char* d_input, * d_output;
  int* d_histogram, * d_sum;

  CSC(cudaMalloc(&d_input, n * sizeof(unsigned char)));
  CSC(cudaMalloc(&d_output, n * sizeof(unsigned char)));
  CSC(cudaMalloc(&d_histogram, 256 * sizeof(int)));
  CSC(cudaMalloc(&d_sum, 256 * sizeof(int)));
  CSC(cudaMemset(d_histogram, 0, 256 * sizeof(int)));

  CSC(cudaMemcpy(d_input, input, n * sizeof(unsigned char), cudaMemcpyHostToDevice));

  compute_histogram <<<256, 256>>> (d_input, d_histogram, n);
  CSC(cudaGetLastError());
  CSC(cudaDeviceSynchronize());

  compute_sum <<<1, 256 >>> (d_histogram, d_sum);
  CSC(cudaGetLastError());
  CSC(cudaDeviceSynchronize());

  kernel_sort <<<256, 256>>> (d_input, d_output, d_sum, n);
  CSC(cudaGetLastError());
  CSC(cudaDeviceSynchronize());

  CSC(cudaMemcpy(output, d_output, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  CSC(cudaFree(d_input));
  CSC(cudaFree(d_output));
  CSC(cudaFree(d_histogram));
  CSC(cudaFree(d_sum));

  fwrite(output, sizeof(unsigned char), n, stdout);

  free(input);
  free(output);
  return 0;
}
