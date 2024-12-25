#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#include <cmath>
#include <device_launch_parameters.h>
using namespace std;

const int TILE_WIDTH = 16;  

// 使用共享内存的矩阵乘法核函数
__global__ void MatrixMulSharedMemKernel(float *A, float *B, float *C, int wA, int wB) {
    // 块和线程的索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 初始化Csub变量
    float Csub = 0;

    // 计算A和B的子矩阵
    for (int a = wA * TILE_WIDTH * by, b = TILE_WIDTH * bx; a < wA * TILE_WIDTH * by + wA - 1 && b < TILE_WIDTH * bx + wB - 1; a += TILE_WIDTH, b += TILE_WIDTH) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH + 1];  // 增加1以减少银行冲突
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 1];  // 增加1以减少银行冲突

        // 加载子矩阵A到共享内存
        int aRow = a / wA + ty;
        int aCol = a % wA + tx;
        if (aRow < wA && aCol < wA)
            As[ty][tx] = A[aRow * wA + aCol];
        else
            As[ty][tx] = 0.0f;

        // 加载子矩阵B到共享内存
        int bRow = b / wB + ty;
        int bCol = b % wB + tx;
        if (bRow < wA && bCol < wB)
            Bs[ty][tx] = B[bRow * wB + bCol];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();  // 在加载完子矩阵后同步

        // 进行矩阵乘法
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // 确保计算完成后再加载下一个子矩阵
    }

    // 存储结果到矩阵C
    int row_C = by * TILE_WIDTH + ty;
    int col_C = bx * TILE_WIDTH + tx;
    if (row_C < wA && col_C < wB)
        C[row_C * wB + col_C] = Csub;
}

// CPU端的矩阵乘法计算
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB) {
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

// 比较CPU和GPU计算结果的差异
void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol) {
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i, j, k;
    int error_count = 0;
    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i++) {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) {
                if (error_count < iListLength) {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    printf(" \n  Total Errors = %d\n", error_count);
}

// 解析命令行参数
void getArg(int argc, char *argv[], int &size, int &check) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <check_enable> <size>\n";
        cerr << "\tcheck_enable: 1 to enable result checking\n";
        cerr << "\tsize: size of the matrix\n";
        exit(1);
    }
    int val1, val2;
    try {
        val1 = stoi(argv[1]);
        val2 = stoi(argv[2]);
    } catch (const invalid_argument &e) {
        cerr << "ERROR: parameters should be integer\n";
        exit(1);
    }
    check = val1;
    size = val2;
}

int main(int argc, char *argv[]) {
    int size, check;
    getArg(argc, argv, size, check);

    int m = size, n = size, k = size;

    // 声明存放在GPU上的数组
    float *h_M, *h_N, *d_M, *d_N;
    float *h_P, *d_P;
    size_t sizeM = m * k * sizeof(float);
    size_t sizeN = k * n * sizeof(float);
    size_t sizeP = m * n * sizeof(float);

    // Allocate host memory
    h_M = (float *)malloc(sizeM);
    h_N = (float *)malloc(sizeN);
    h_P = (float *)malloc(sizeP);
    float *reference = (float *)malloc(sizeP);

    // Allocate device memory
    cudaMalloc(&d_M, sizeM);
    cudaMalloc(&d_N, sizeN);
    cudaMalloc(&d_P, sizeP);

    // Init data
    for (int i = 0; i < m * n; ++i) {
        if (i % 2 == 0)
            h_M[i] = 1.0;
        else
            h_M[i] = 0.5;
    }

    for (int i = 0; i < n * k; ++i) {
        if (i % 2 == 0)
            h_N[i] = 0.5;
        else
            h_N[i] = 1.0;
    }

    // Copy data from CPU to GPU
    cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);

    // Timing records
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel 定义grid&block
    dim3 grid((int)ceil(k * 1.0 / TILE_WIDTH), (int)ceil(m * 1.0 / TILE_WIDTH));
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    int nIter = 5;
#ifdef USE_CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
#endif
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int j = 0; j < nIter; j++) {
        // 你可以启用这个来使用CPU计算参考值
        // matrixMulCPU(reference, h_M, h_N, m, k, n);
        // 使用优化后的GPU矩阵乘法
        MatrixMulSharedMemKernel<<<grid, block>>>(d_M, d_N, d_P, m, n);
        // 使用cublasSgemm函数来计算矩阵乘法（可选）
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_N, n, d_M, k, &beta, d_P, n);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float msecPerMatrixMul;
    cudaEventElapsedTime(&msecPerMatrixMul, start, stop);
    msecPerMatrixMul /= nIter;
    printf("Kernel Elapsed Time: %.3f ms\n", msecPerMatrixMul);

    // 计算和打印性能
    double flopsPerMatrixMul = 2.0 * (double)m * (double)n * (double)k;
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s\n", gigaFlops);

    // Copy result back to host memory
    cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost);

    // Optionally check results
    if (check) {
        matrixMulCPU(reference, h_M, h_N, m, k, n);
        printDiff(reference, h_P, n, m, 10, 1e-5);
    }

    // Clean up
    free(h_M);
    free(h_N);
    free(h_P);
    free(reference);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
