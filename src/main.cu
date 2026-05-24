#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS Error, status = " << status                   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ============================================================
// CPU reference: row-major
// C[M, N] = A[M, K] * B[K, N]
// ============================================================
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================
// naive GEMM
// one thread computes one C element
// ============================================================
__global__ void gemm_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N direction
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M direction

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ============================================================
// shared memory blocktile GEMM
// block computes 16x16 C tile
// ============================================================
#define TILE 16

__global__ void gemm_blocktile_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float AS[TILE][TILE];
    __shared__ float BS[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    float sum = 0.0f;

    for (int r = 0; r < CEIL(K, TILE); r++) {
        int a_col = r * TILE + tx;
        int b_row = r * TILE + ty;

        if (row < M && a_col < K) {
            AS[ty][tx] = A[row * K + a_col];
        } else {
            AS[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            BS[ty][tx] = B[b_row * N + col];
        } else {
            BS[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += AS[ty][k] * BS[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// 2x2 thread tile GEMM
// block computes 32x32 C tile
// each thread computes 2x2 C elements
// ============================================================
#define BM 32
#define BN 32
#define BK 16
#define TM 2
#define TN 2

__global__ void gemm_threadtile2x2_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float AS[BM][BK];
    __shared__ float BS[BK][BN];

    int tx = threadIdx.x; // 0 ~ 15
    int ty = threadIdx.y; // 0 ~ 15

    int row_base = blockIdx.y * BM + ty * TM;
    int col_base = blockIdx.x * BN + tx * TN;

    float sum[TM][TN] = {0.0f};

    for (int r = 0; r < CEIL(K, BK); r++) {
        int a_col = r * BK + tx;

        for (int i = 0; i < TM; i++) {
            int a_row = row_base + i;

            if (a_row < M && a_col < K) {
                AS[ty * TM + i][tx] = A[a_row * K + a_col];
            } else {
                AS[ty * TM + i][tx] = 0.0f;
            }
        }

        int b_row = r * BK + ty;

        for (int j = 0; j < TN; j++) {
            int b_col = col_base + j;

            if (b_row < K && b_col < N) {
                BS[ty][tx * TN + j] = B[b_row * N + b_col];
            } else {
                BS[ty][tx * TN + j] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) {
                float a = AS[ty * TM + i][k];

                for (int j = 0; j < TN; j++) {
                    sum[i][j] += a * BS[k][tx * TN + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int row = row_base + i;

        if (row < M) {
            for (int j = 0; j < TN; j++) {
                int col = col_base + j;

                if (col < N) {
                    C[row * N + col] = sum[i][j];
                }
            }
        }
    }
}

// ============================================================
// 4x4 thread tile GEMM
// block computes 64x64 C tile
// each thread computes 4x4 C elements
// ============================================================
#define BM4 64
#define BN4 64
#define BK4 16
#define TM4 4
#define TN4 4

__global__ void gemm_threadtile4x4_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float AS[BM4][BK4];
    __shared__ float BS[BK4][BN4];

    int tx = threadIdx.x; // 0 ~ 15
    int ty = threadIdx.y; // 0 ~ 15

    int row_base = blockIdx.y * BM4 + ty * TM4;
    int col_base = blockIdx.x * BN4 + tx * TN4;

    float sum[TM4][TN4];

#pragma unroll
    for (int i = 0; i < TM4; i++) {
#pragma unroll
        for (int j = 0; j < TN4; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int r = 0; r < CEIL(K, BK4); r++) {
        // load A tile: BM4 x BK4 = 64 x 16
        // 共有 1024 个 A 元素，256 个线程，每线程加载 4 个
        int a_col = r * BK4 + tx;

#pragma unroll
        for (int i = 0; i < TM4; i++) {
            int a_row = row_base + i;

            if (a_row < M && a_col < K) {
                AS[ty * TM4 + i][tx] = A[a_row * K + a_col];
            } else {
                AS[ty * TM4 + i][tx] = 0.0f;
            }
        }

        // load B tile: BK4 x BN4 = 16 x 64
        // 共有 1024 个 B 元素，256 个线程，每线程加载 4 个
        int b_row = r * BK4 + ty;

#pragma unroll
        for (int j = 0; j < TN4; j++) {
            int b_col = col_base + j;

            if (b_row < K && b_col < N) {
                BS[ty][tx * TN4 + j] = B[b_row * N + b_col];
            } else {
                BS[ty][tx * TN4 + j] = 0.0f;
            }
        }

        __syncthreads();

        // compute 4x4 outputs
#pragma unroll
        for (int k = 0; k < BK4; k++) {
            float a_frag[TM4];
            float b_frag[TN4];

#pragma unroll
            for (int i = 0; i < TM4; i++) {
                a_frag[i] = AS[ty * TM4 + i][k];
            }

#pragma unroll
            for (int j = 0; j < TN4; j++) {
                b_frag[j] = BS[k][tx * TN4 + j];
            }

#pragma unroll
            for (int i = 0; i < TM4; i++) {
#pragma unroll
                for (int j = 0; j < TN4; j++) {
                    sum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // write back C
#pragma unroll
    for (int i = 0; i < TM4; i++) {
        int row = row_base + i;

        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN4; j++) {
                int col = col_base + j;

                if (col < N) {
                    C[row * N + col] = sum[i][j];
                }
            }
        }
    }
}

float benchmark_threadtile4x4(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    dim3 block(16, 16);
    dim3 grid(CEIL(N, BN4), CEIL(M, BM4));

    for (int i = 0; i < warmup; i++) {
        gemm_threadtile4x4_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        gemm_threadtile4x4_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
// utils
// ============================================================
void init_random(std::vector<float>& x, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : x) {
        v = dist(gen);
    }
}

float max_error(const std::vector<float>& ref, const std::vector<float>& out) {
    float err = 0.0f;

    for (size_t i = 0; i < ref.size(); i++) {
        err = std::max(err, std::abs(ref[i] - out[i]));
    }

    return err;
}

double calc_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    return flops / (ms / 1000.0) / 1e9;
}

// ============================================================
// cuBLAS row-major wrapper
//
// Our row-major GEMM:
// C[M,N] = A[M,K] * B[K,N]
//
// cuBLAS assumes column-major.
// Equivalent:
// C^T[N,M] = B^T[N,K] * A^T[K,M]
// ============================================================
void launch_cublas_sgemm(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        d_B,
        N,
        d_A,
        K,
        &beta,
        d_C,
        N
    ));
}

// ============================================================
// benchmark functions
// ============================================================
float benchmark_naive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    dim3 block(16, 16);
    dim3 grid(CEIL(N, block.x), CEIL(M, block.y));

    for (int i = 0; i < warmup; i++) {
        gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}

float benchmark_blocktile(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    dim3 block(TILE, TILE);
    dim3 grid(CEIL(N, TILE), CEIL(M, TILE));

    for (int i = 0; i < warmup; i++) {
        gemm_blocktile_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        gemm_blocktile_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}

float benchmark_threadtile2x2(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    dim3 block(16, 16);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));

    for (int i = 0; i < warmup; i++) {
        gemm_threadtile2x2_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        gemm_threadtile2x2_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}

float benchmark_cublas(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; i++) {
        launch_cublas_sgemm(handle, d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        launch_cublas_sgemm(handle, d_A, d_B, d_C, M, N, K);
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

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_ref(size_C, 0.0f);
    std::vector<float> h_C_naive(size_C, 0.0f);
    std::vector<float> h_C_blocktile(size_C, 0.0f);
    std::vector<float> h_C_threadtile2x2(size_C, 0.0f);
    std::vector<float> h_C_threadtile4x4(size_C, 0.0f);
    std::vector<float> h_C_cublas(size_C, 0.0f);

    init_random(h_A, 123);
    init_random(h_B, 456);

    if (do_cpu_check) {
        std::cout << "Running CPU reference..." << std::endl;
        gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int warmup = 10;
    int repeat = 50;

    float ms_naive = benchmark_naive(d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float ms_blocktile = benchmark_blocktile(d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_blocktile.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float ms_threadtile2x2 = benchmark_threadtile2x2(d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_threadtile2x2.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float ms_threadtile4x4 = benchmark_threadtile4x4(d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_threadtile4x4.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float ms_cublas = benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, warmup, repeat);
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "naive      : " << ms_naive
              << " ms, " << calc_gflops(M, N, K, ms_naive) << " GFLOPS" << std::endl;

    std::cout << "blocktile  : " << ms_blocktile
              << " ms, " << calc_gflops(M, N, K, ms_blocktile) << " GFLOPS" << std::endl;

    std::cout << "thread2x2  : " << ms_threadtile2x2
          << " ms, " << calc_gflops(M, N, K, ms_threadtile2x2) << " GFLOPS" << std::endl;

    std::cout << "thread4x4  : " << ms_threadtile4x4
            << " ms, " << calc_gflops(M, N, K, ms_threadtile4x4) << " GFLOPS" << std::endl;

    std::cout << "cuBLAS     : " << ms_cublas
              << " ms, " << calc_gflops(M, N, K, ms_cublas) << " GFLOPS" << std::endl;

    if (do_cpu_check) {
        std::cout << "max error naive     : " << max_error(h_C_ref, h_C_naive) << std::endl;
        std::cout << "max error blocktile : " << max_error(h_C_ref, h_C_blocktile) << std::endl;
        std::cout << "max error thread2x2 : " << max_error(h_C_ref, h_C_threadtile2x2) << std::endl;
        std::cout << "max error thread4x4 : " << max_error(h_C_ref, h_C_threadtile4x4) << std::endl;
        std::cout << "max error cuBLAS    : " << max_error(h_C_ref, h_C_cublas) << std::endl;
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

    run_one_shape(1, 4096, 4096, false);
    run_one_shape(16, 4096, 4096, false);
    run_one_shape(32, 4096, 4096, false);
    run_one_shape(128, 4096, 4096, false);

    run_one_shape(32, 11008, 4096, false);
    run_one_shape(32, 4096, 11008, false);

    return 0;
}