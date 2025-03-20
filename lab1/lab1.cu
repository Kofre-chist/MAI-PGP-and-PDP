#include <stdio.h>

#define CSC(call)       \
do {                    \
    cudaError_t status = call;          \
    if  (status != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));   \
        exit(0);                        \
    }                                   \
} while (0)

__global__ void kernel(double *vec1, double *vec2, double *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while (idx < n) {
        result[idx] = vec1[idx] > vec2[idx] ? vec1[idx] : vec2[idx];
        idx += offset;
    }
}

int main() {
    long int n;
    scanf("%ld", &n);

    double *vec1 = (double *)malloc(sizeof(double) * n);
    double *vec2 = (double *)malloc(sizeof(double) * n);
    double *result = (double *)malloc(sizeof(double) * n);
    
    for (int i = 0; i < n; ++i){
      scanf("%lf", &vec1[i]);
    }

    for (int i = 0; i < n; ++i){
      scanf("%lf", &vec2[i]);
    }

    double *dev_vec1, *dev_vec2, *dev_result;

    CSC(cudaMalloc(&dev_vec1, sizeof(double) * n));
    CSC(cudaMalloc(&dev_vec2, sizeof(double) * n));
    CSC(cudaMalloc(&dev_result, sizeof(double) * n));
    CSC(cudaMemcpy(dev_vec1, vec1, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_vec2, vec2, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_result, result, sizeof(double) * n, cudaMemcpyHostToDevice));

    kernel<<<1024, 1024>>>(dev_vec1, dev_vec2, dev_result, n);

    CSC(cudaMemcpy(result, dev_result, sizeof(double) * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%.10lf ", result[i]);
    }
    printf("\n");

    free(vec1);
    free(vec2);
    free(result);
    CSC(cudaFree(dev_vec1));
    CSC(cudaFree(dev_vec2));
    CSC(cudaFree(dev_result));

    return 0;
}
