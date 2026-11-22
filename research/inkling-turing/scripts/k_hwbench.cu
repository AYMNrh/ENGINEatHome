// Measured hardware ceilings for THEORY.md: device copy bandwidth, read-only streaming bandwidth, cuBLAS FP16 and INT8 GEMM.
// nvcc -O3 -arch=sm_75 -o hwbench k_hwbench.cu -lcublas
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS %d line %d\n", (int) s, __LINE__); return 1; } } while (0)

__global__ void read_kernel(const int4 * __restrict__ p, size_t n, int * out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = __ldg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *out = 1;
}

int main() {
    const size_t bytes = 2ull << 30; // 2 GiB
    char *a, *b; int * flag;
    CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMalloc(&flag, 4));
    CK(cudaMemset(a, 1, bytes)); CK(cudaMemset(b, 0, bytes));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms;

    // device-to-device copy (reads + writes 2 GiB each)
    CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice));
    float best = 1e9;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(e0); CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); best = ms < best ? ms : best;
    }
    printf("copy_bandwidth_GBs %.1f\n", 2.0 * bytes / (best * 1e-3) / 1e9);

    // read-only stream (what decode does with weights)
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    const int blocks = prop.multiProcessorCount * 8;
    read_kernel<<<blocks, 256>>>((const int4 *) a, bytes / 16, flag);
    best = 1e9;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(e0); read_kernel<<<blocks, 256>>>((const int4 *) a, bytes / 16, flag); cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); best = ms < best ? ms : best;
    }
    printf("read_bandwidth_GBs %.1f\n", bytes / (best * 1e-3) / 1e9);

    // small read (what one decode kernel sees: ~16 MB, partly L2 resident across reps -> report as upper bound)
    const size_t small = 16ull << 20;
    best = 1e9;
    for (int r = 0; r < 20; ++r) {
        const char * base = a + (size_t) (r % 64) * small;    // different buffers each rep, no L2 reuse
        cudaEventRecord(e0); read_kernel<<<blocks, 256>>>((const int4 *) base, small / 16, flag); cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); best = ms < best ? ms : best;
    }
    printf("read16MB_bandwidth_GBs %.1f\n", small / (best * 1e-3) / 1e9);
    CK(cudaFree(b));

    cublasHandle_t h; CB(cublasCreate(&h));
    for (int n : {4096, 8192}) {
        half *A, *B, *C;
        CK(cudaMalloc(&A, (size_t) n*n*2)); CK(cudaMalloc(&B, (size_t) n*n*2)); CK(cudaMalloc(&C, (size_t) n*n*2));
        CK(cudaMemset(A, 0, (size_t) n*n*2)); CK(cudaMemset(B, 0, (size_t) n*n*2));
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CB(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n));
        best = 1e9;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0); CB(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n)); cudaEventRecord(e1); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1); best = ms < best ? ms : best;
        }
        printf("fp16_gemm_%d_TFLOPS %.1f\n", n, 2.0 * n * n * (double) n / (best * 1e-3) / 1e12);
        cudaFree(A); cudaFree(B); cudaFree(C);

        int8_t *Ai, *Bi; int32_t * Ci;
        CK(cudaMalloc(&Ai, (size_t) n*n)); CK(cudaMalloc(&Bi, (size_t) n*n)); CK(cudaMalloc(&Ci, (size_t) n*n*4));
        CK(cudaMemset(Ai, 0, (size_t) n*n)); CK(cudaMemset(Bi, 0, (size_t) n*n));
        const int32_t ai = 1, bi = 0;
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &ai, Ai, CUDA_R_8I, n, Bi, CUDA_R_8I, n, &bi, Ci, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        best = 1e9;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0);
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &ai, Ai, CUDA_R_8I, n, Bi, CUDA_R_8I, n, &bi, Ci, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1); best = ms < best ? ms : best;
        }
        printf("int8_gemm_%d_TOPS %.1f\n", n, 2.0 * n * n * (double) n / (best * 1e-3) / 1e12);
        cudaFree(Ai); cudaFree(Bi); cudaFree(Ci);
    }
    int sm, mem; cudaDeviceGetAttribute(&sm, cudaDevAttrClockRate, dev); cudaDeviceGetAttribute(&mem, cudaDevAttrMemoryClockRate, dev);
    printf("sms %d sm_clock_max_MHz %d mem_clock_MHz %d bus_bits %d l2_MB %.1f\n", prop.multiProcessorCount, sm/1000, mem/1000, prop.memoryBusWidth, prop.l2CacheSize/1048576.0);
    return 0;
}
