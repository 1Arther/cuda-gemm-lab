#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS Error, status = " << status                    \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================
// Minimal WMMA GEMM
//
// A: half  [M, K] row-major
// B: half  [K, N] row-major
// C: float [M, N] row-major
//
// one warp computes one 16x16 C tile
// ============================================================
__global__ void wmma_gemm_kernel(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K
) {
    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    int row = tile_row * WMMA_M;
    int col = tile_col * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const half* a_tile = A + row * K + k0;
        const half* b_tile = B + k0 * N + col;

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + row * N + col;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// 4-warp WMMA GEMM
//
// 一个 block = 4 warps = 128 threads
// 一个 block 计算 32 x 32 C tile
// 每个 warp 计算一个 16 x 16 C tile
// ============================================================
#define WM_WARP_M 2
#define WM_WARP_N 2
#define WM_BLOCK_WARP 4

#define WM_BLOCK_M (WM_WARP_M * WMMA_M)  // 32
#define WM_BLOCK_N (WM_WARP_N * WMMA_N)  // 32

__global__ void wmma_gemm4warp_kernel(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K
) {
    int tid = threadIdx.x;
    int warpId = tid / warpSize;

    if (warpId >= WM_BLOCK_WARP) {
        return;
    }

    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    int block_row = tile_row * WM_BLOCK_M;
    int block_col = tile_col * WM_BLOCK_N;

    int warp_m = warpId / WM_WARP_N;
    int warp_n = warpId % WM_WARP_N;

    int row = block_row + warp_m * WMMA_M;
    int col = block_col + warp_n * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const half* a_tile = A + row * K + k0;
        const half* b_tile = B + k0 * N + col;

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + row * N + col;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}
// ============================================================
// CPU reference
// A/B: half
// C: float
// ============================================================
void gemm_cpu_ref(
    const std::vector<half>& A,
    const std::vector<half>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                float a = __half2float(A[i * K + k]);
                float b = __half2float(B[k * N + j]);
                sum += a * b;
            }

            C[i * N + j] = sum;
        }
    }
}

void init_random_half(std::vector<half>& x, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : x) {
        v = __float2half(dist(gen));
    }
}

void calc_error(
    const std::vector<float>& ref,
    const std::vector<float>& out,
    float& max_err,
    double& mean_err
) {
    max_err = 0.0f;
    mean_err = 0.0;

    for (size_t i = 0; i < ref.size(); i++) {
        float err = std::abs(ref[i] - out[i]);
        max_err = std::max(max_err, err);
        mean_err += err;
    }

    mean_err /= ref.size();
}

double calc_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    return flops / (ms / 1000.0) / 1e9;
}

// ============================================================
// WMMA benchmark
// ============================================================
float benchmark_wmma(
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    dim3 block(32);  // one warp per block
    dim3 grid(N / WMMA_N, M / WMMA_M);

    for (int i = 0; i < warmup; i++) {
        wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}

float benchmark_wmma4warp(
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    dim3 block(32 * WM_BLOCK_WARP);  // 4 warps = 128 threads
    dim3 grid(N / WM_BLOCK_N, M / WM_BLOCK_M);

    for (int i = 0; i < warmup; i++) {
        wmma_gemm4warp_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        wmma_gemm4warp_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}
// ============================================================
// cuBLAS FP16 input + FP32 accumulation baseline
//
// row-major C = A * B
// converted to column-major:
// C^T = B^T * A^T
// ============================================================
void launch_cublas_hgemm_fp32acc(
    cublasHandle_t handle,
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M,
    int N,
    int K
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,              // m
        M,              // n
        K,              // k
        &alpha,
        d_B,
        CUDA_R_16F,
        N,              // lda
        d_A,
        CUDA_R_16F,
        K,              // ldb
        &beta,
        d_C,
        CUDA_R_32F,
        N,              // ldc
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

float benchmark_cublas(
    cublasHandle_t handle,
    const half* d_A,
    const half* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; i++) {
        launch_cublas_hgemm_fp32acc(handle, d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        launch_cublas_hgemm_fp32acc(handle, d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}

// ============================================================
// run one shape
// ============================================================
void run_one_shape(int M, int N, int K, bool do_cpu_check) {
    std::cout << "\n==============================" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        std::cout << "Skip: M/N/K must be multiples of 16 for this minimal WMMA kernel." << std::endl;
        return;
    }

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::vector<half> h_A(size_A);
    std::vector<half> h_B(size_B);
    std::vector<float> h_C_ref(size_C, 0.0f);
    std::vector<float> h_C_wmma(size_C, 0.0f);
    std::vector<float> h_C_wmma4warp(size_C, 0.0f);
    std::vector<float> h_C_cublas(size_C, 0.0f);

    init_random_half(h_A, 123);
    init_random_half(h_B, 456);

    if (do_cpu_check) {
        std::cout << "Running CPU reference..." << std::endl;
        gemm_cpu_ref(h_A, h_B, h_C_ref, M, N, K);
    }

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int warmup = 10;
    int repeat = 50;

    float ms_wmma = benchmark_wmma(d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_wmma.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float ms_wmma4warp = benchmark_wmma4warp(d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_wmma4warp.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float ms_cublas = benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "wmma minimal : " << ms_wmma
              << " ms, " << calc_gflops(M, N, K, ms_wmma)
              << " GFLOPS" << std::endl;

    std::cout << "wmma 4warp   : " << ms_wmma4warp
              << " ms, " << calc_gflops(M, N, K, ms_wmma4warp)
              << " GFLOPS" << std::endl;

    std::cout << "cuBLAS hgemm : " << ms_cublas
              << " ms, " << calc_gflops(M, N, K, ms_cublas)
              << " GFLOPS" << std::endl;

    if (do_cpu_check) {
        float max_err_wmma = 0.0f;
        double mean_err_wmma = 0.0;
        calc_error(h_C_ref, h_C_wmma, max_err_wmma, mean_err_wmma);

        float max_err_wmma4warp = 0.0f;
        double mean_err_wmma4warp = 0.0;
        calc_error(h_C_ref, h_C_wmma4warp, max_err_wmma4warp, mean_err_wmma4warp);

        float max_err_cublas = 0.0f;
        double mean_err_cublas = 0.0;
        calc_error(h_C_ref, h_C_cublas, max_err_cublas, mean_err_cublas);

        std::cout << "max error wmma      : " << max_err_wmma << std::endl;
        std::cout << "mean error wmma     : " << mean_err_wmma << std::endl;

        std::cout << "max error wmma4warp : " << max_err_wmma4warp << std::endl;
        std::cout << "mean error wmma4warp: " << mean_err_wmma4warp << std::endl;

        std::cout << "max error cuBLAS    : " << max_err_cublas << std::endl;
        std::cout << "mean error cuBLAS   : " << mean_err_cublas << std::endl;
    } else {
        std::cout << "CPU check skipped for large shape." << std::endl;
    }

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

int main() {
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << std::endl;

    run_one_shape(256, 256, 256, true);
    run_one_shape(512, 512, 512, false);
    run_one_shape(1024, 1024, 1024, false);
    run_one_shape(2048, 2048, 2048, false);

    return 0;
}