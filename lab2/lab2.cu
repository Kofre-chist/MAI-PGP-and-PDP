#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4* out, int w, int h, int wn, int hn) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;

  int step_x = w / wn;
  int step_y = h / hn;
  int size = step_x * step_y;

  for (int y = idy; y < hn; y += offsety) {
    for (int x = idx; x < wn; x += offsetx) {

      float4 blockAcc = make_float4(0, 0, 0, 0);

      for (int j = 0; j < step_y; ++j) {
        for (int i = 0; i < step_x; ++i) {
          uchar4 p = tex2D<uchar4>(tex, (x * step_x + i + 0.5f) / w, (y * step_y + j + 0.5f) / h);
          blockAcc.x += p.x;
          blockAcc.y += p.y;
          blockAcc.z += p.z;
          blockAcc.w += p.w;
        }
      }

      out[y * wn + x] = make_uchar4(blockAcc.x / size, blockAcc.y / size, blockAcc.z / size, blockAcc.w / size);
    }
  }
}

int main() {
  char input_file[256], output_file[256];
  int wn, hn;

  scanf("%s", input_file);
  scanf("%s", output_file);
  scanf("%d %d", &wn, &hn);

  int w, h;
  FILE* fp = fopen(input_file, "rb");
  fread(&w, sizeof(int), 1, fp);
  fread(&h, sizeof(int), 1, fp);
  uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
  fread(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  cudaArray* arr;
  cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
  CSC(cudaMallocArray(&arr, &ch, w, h));
  CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = arr;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = true;

  cudaTextureObject_t tex = 0;
  CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

  uchar4* dev_out;
  CSC(cudaMalloc(&dev_out, sizeof(uchar4) * wn * hn));

  kernel << < dim3(16, 16), dim3(32, 32) >> > (tex, dev_out, w, h, wn, hn);
  CSC(cudaGetLastError());

  uchar4* output_data = (uchar4*)malloc(sizeof(uchar4) * wn * hn);
  CSC(cudaMemcpy(output_data, dev_out, sizeof(uchar4) * wn * hn, cudaMemcpyDeviceToHost));

  CSC(cudaDestroyTextureObject(tex));
  CSC(cudaFreeArray(arr));
  CSC(cudaFree(dev_out));

  fp = fopen(output_file, "wb");
  fwrite(&wn, sizeof(int), 1, fp);
  fwrite(&hn, sizeof(int), 1, fp);
  fwrite(output_data, sizeof(uchar4), wn * hn, fp);
  fclose(fp);

  free(data);
  free(output_data);
  return 0;
}
