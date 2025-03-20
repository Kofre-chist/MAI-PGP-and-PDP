#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call)                                               \
do {                                                            \
    cudaError_t status = call;                                  \
    if (status != cudaSuccess) {                                \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(status));\
        exit(0);                                                \
    }                                                           \
} while (0)

#define MAX_CLASSES 32
#define DIM 3

typedef struct {
    int x, y;
} Point;

void computeAVG(uchar4 *pixels, int width, int height, Point *points, int np, double avg[DIM]) {
    avg[0] = 0.0;
    avg[1] = 0.0;
    avg[2] = 0.0;

    for (int i = 0; i < np; ++i) {
        uchar4 p = pixels[points[i].x + points[i].y * width];
        avg[0] += p.x;
        avg[1] += p.y;
        avg[2] += p.z;
    }

    avg[0] /= np;
    avg[1] /= np;
    avg[2] /= np;
}

void computeCovMatrix(uchar4 *pixels, int width, int height, Point *points, int np, double avg[DIM], double variance[DIM][DIM]) {
    variance[0][0] = variance[0][1] = variance[0][2] = 0.0;
    variance[1][0] = variance[1][1] = variance[1][2] = 0.0;
    variance[2][0] = variance[2][1] = variance[2][2] = 0.0;

    for (int i = 0; i < np; ++i) {
        uchar4 p = pixels[points[i].x + points[i].y * width];
        double dx = p.x - avg[0];
        double dy = p.y - avg[1];
        double dz = p.z - avg[2];

        variance[0][0] += dx * dx;
        variance[0][1] += dx * dy;
        variance[0][2] += dx * dz;
        variance[1][0] += dy * dx;
        variance[1][1] += dy * dy;
        variance[1][2] += dy * dz;
        variance[2][0] += dz * dx;
        variance[2][1] += dz * dy;
        variance[2][2] += dz * dz;
    }

    double normFactor = 1.0 / (np - 1);
    variance[0][0] *= normFactor;
    variance[0][1] *= normFactor;
    variance[0][2] *= normFactor;
    variance[1][0] *= normFactor;
    variance[1][1] *= normFactor;
    variance[1][2] *= normFactor;
    variance[2][0] *= normFactor;
    variance[2][1] *= normFactor;
    variance[2][2] *= normFactor;
}

double computeDeterminant(double matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

double computeLogDeterminant(double variance[DIM][DIM]) {
    return log(fabs(computeDeterminant(variance)));
}

void computeInvMatrix(double variance[DIM][DIM], double inverse[DIM][DIM]) {
    double determinant = computeDeterminant(variance);

    double invDet = 1.0 / determinant;
    inverse[0][0] = (variance[1][1] * variance[2][2] - variance[1][2] * variance[2][1]) * invDet;
    inverse[0][1] = (variance[0][2] * variance[2][1] - variance[0][1] * variance[2][2]) * invDet;
    inverse[0][2] = (variance[0][1] * variance[1][2] - variance[0][2] * variance[1][1]) * invDet;

    inverse[1][0] = (variance[1][2] * variance[2][0] - variance[1][0] * variance[2][2]) * invDet;
    inverse[1][1] = (variance[0][0] * variance[2][2] - variance[0][2] * variance[2][0]) * invDet;
    inverse[1][2] = (variance[0][2] * variance[1][0] - variance[0][0] * variance[1][2]) * invDet;

    inverse[2][0] = (variance[1][0] * variance[2][1] - variance[1][1] * variance[2][0]) * invDet;
    inverse[2][1] = (variance[0][1] * variance[2][0] - variance[0][0] * variance[2][1]) * invDet;
    inverse[2][2] = (variance[0][0] * variance[1][1] - variance[0][1] * variance[1][0]) * invDet;
}

struct ConstMem {
    double classMeanColors[MAX_CLASSES][DIM];
    double classCovInv[MAX_CLASSES][DIM][DIM];
    double classLogDet[MAX_CLASSES];
};
__constant__ ConstMem constMem;

__device__ double calculatePropability(uchar4 p, int classIdx) {
    return  (-(p.x - constMem.classMeanColors[classIdx][0])) * 
                ((p.x - constMem.classMeanColors[classIdx][0]) * constMem.classCovInv[classIdx][0][0] + 
                 (p.y - constMem.classMeanColors[classIdx][1]) * constMem.classCovInv[classIdx][1][0] + 
                 (p.z - constMem.classMeanColors[classIdx][2]) * constMem.classCovInv[classIdx][2][0]) +
            (-(p.y - constMem.classMeanColors[classIdx][1])) * 
                ((p.x - constMem.classMeanColors[classIdx][0]) * constMem.classCovInv[classIdx][0][1] + 
                 (p.y - constMem.classMeanColors[classIdx][1]) * constMem.classCovInv[classIdx][1][1] + 
                 (p.z - constMem.classMeanColors[classIdx][2]) * constMem.classCovInv[classIdx][2][1]) +
            (-(p.z - constMem.classMeanColors[classIdx][2])) * 
                ((p.x - constMem.classMeanColors[classIdx][0]) * constMem.classCovInv[classIdx][0][2] + 
                 (p.y - constMem.classMeanColors[classIdx][1]) * constMem.classCovInv[classIdx][1][2] + 
                 (p.z - constMem.classMeanColors[classIdx][2]) * constMem.classCovInv[classIdx][2][2]) -
            constMem.classLogDet[classIdx];
}

__device__ int determinePixelClass(uchar4 p, int nc) {
    double maxScore = -524288;
    int result = 0;
    for (int i = 0; i < nc; ++i){
        double curScore = calculatePropability(p, i);
        if (curScore > maxScore){
          maxScore = curScore;
          result = i;
        }
    }
    return result;
}

__global__ void kernel(uchar4 *pixels, int width, int height, int nc) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offsetx = blockDim.x * gridDim.x;
    int size = width * height;
    while (idx < size) {
        pixels[idx].w = determinePixelClass(pixels[idx], nc);
        idx += offsetx;
    }
}

int main() {
    char input_file[256], output_file[256];
    int nc;

    scanf("%s", input_file);
    scanf("%s", output_file);
    scanf("%d", &nc);

    Point **pixelPoints = (Point **)malloc(nc * sizeof(Point *));
    int *np = (int *) malloc(nc * sizeof(int));

    if (pixelPoints == NULL || np == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < nc; ++i) {
        scanf("%d", &np[i]);
        pixelPoints[i] = (Point *)malloc(np[i] * sizeof(Point));

        if (pixelPoints[i] == NULL) {
            fprintf(stderr, "Memory allocation failed!\n");
            return 1;
        }

        for (int j = 0; j < np[i]; ++j) {
            scanf("%d %d", &pixelPoints[i][j].x, &pixelPoints[i][j].y);
        }
    }

    int width, height;
    FILE *infile = fopen(input_file, "rb");
    fread(&width, sizeof(int), 1, infile);
    fread(&height, sizeof(int), 1, infile);
    uchar4 *data = (uchar4 *)malloc(width * height * sizeof(uchar4));
    if (data == NULL) {
      fprintf(stderr, "Memory allocation failed!\n");
      return 1;
    }
    fread(data, sizeof(uchar4), width * height, infile);
    fclose(infile);

    double avgColors[MAX_CLASSES][DIM];
    double covMatrices[MAX_CLASSES][DIM][DIM];
    double invMatrices[MAX_CLASSES][DIM][DIM];
    double logDetValues[MAX_CLASSES];

    for (int i = 0; i < nc; ++i) {
        computeAVG(data, width, height, pixelPoints[i], np[i], avgColors[i]);
        computeCovMatrix(data, width, height, pixelPoints[i], np[i], avgColors[i], covMatrices[i]);
        computeInvMatrix(covMatrices[i], invMatrices[i]);
        logDetValues[i] = computeLogDeterminant(covMatrices[i]);
    }

    ConstMem hostMem;
    memcpy(hostMem.classMeanColors, avgColors, sizeof(avgColors));
    memcpy(hostMem.classCovInv, invMatrices, sizeof(invMatrices));
    memcpy(hostMem.classLogDet, logDetValues, sizeof(logDetValues));

    CSC(cudaMemcpyToSymbol(constMem, &hostMem, sizeof(ConstMem)));

    uchar4 *dev_out = NULL;
    CSC(cudaMalloc(&dev_out, width * height * sizeof(uchar4)));
    CSC(cudaMemcpy(dev_out, data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    kernel<<<1024, 1024>>>(dev_out, width, height, nc);
    CSC(cudaDeviceSynchronize());
	  CSC(cudaGetLastError());

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    printf("time = %f ms\n", t);

    CSC(cudaMemcpy(data, dev_out, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));

    FILE *outfile = fopen(output_file, "wb");
    fwrite(&width, sizeof(int), 1, outfile);
    fwrite(&height, sizeof(int), 1, outfile);
    fwrite(data, sizeof(uchar4), width * height, outfile);
    fclose(outfile);

    CSC(cudaFree(dev_out));
    for (int i = 0; i < nc; ++i) {
        free(pixelPoints[i]);
    }
    free(pixelPoints);
    free(np);
    free(data);

    return 0;
}
