# CUDA GEMM Lab

这是一个用于学习和分析 CUDA GEMM 优化的实验项目，主要实现并 benchmark 多个 SGEMM kernel。

项目重点不是超过 cuBLAS，而是通过逐步实现不同版本的 GEMM，理解 CUDA 算子优化中的核心思想：

```text
naive GEMM
-> shared memory block tiling
-> 2x2 thread tile / register blocking
-> 4x4 thread tile / register blocking
-> cuBLAS baseline
```

核心关注点包括：

```text
global memory 访问
shared memory 复用
register blocking
thread tile 设计
arithmetic intensity
occupancy 权衡
不同 shape 下的性能差异
cuBLAS baseline 对比
```

---

## 实验环境

Benchmark 环境如下：

```text
GPU: NVIDIA A40
CUDA: Linux CUDA Toolkit / nvcc
Build: CMake + CUDA C++17
Baseline: cuBLAS SGEMM
```

---

## 实现版本

### 1. Naive GEMM

每个 CUDA thread 负责计算一个输出元素：

```cpp
C[row][col] = sum_k A[row][k] * B[k][col]
```

这个版本直接从 global memory 读取 A 和 B，没有显式利用 block 内的数据复用，因此 global memory 重复读取较多。

---

### 2. Shared Memory Block Tiling

每个 block 计算一个 `16 x 16` 的 C tile。

A tile 和 B tile 会先加载到 shared memory：

```text
global memory -> shared memory -> register
```

这样 block 内多个线程可以复用 shared memory 中的 A/B 数据，从而减少 global memory 的重复读取。

---

### 3. 2x2 Thread Tile

每个 thread 负责计算一个 `2 x 2` 的输出小块。

每个线程维护 4 个寄存器累加器：

```cpp
float sum[2][2];
```

相比普通 shared memory block tiling，2x2 thread tile 进一步增强了寄存器级数据复用，提高了计算密度。

---

### 4. 4x4 Thread Tile

每个 thread 负责计算一个 `4 x 4` 的输出小块。

每个线程维护 16 个寄存器累加器：

```cpp
float sum[4][4];
```

这个版本在大矩阵 GEMM 上通常更快，因为 A/B fragment 在寄存器中被复用更多次。但它也会增加寄存器使用量，降低 occupancy，并且在小 M shape 下可能退化。

---

### 5. cuBLAS Baseline

cuBLAS SGEMM 作为工业级 baseline。

由于 cuBLAS 默认使用 column-major 布局，而本项目中矩阵使用 row-major 布局，因此使用如下等价变换：

```text
C[M, N] = A[M, K] * B[K, N]
```

等价于：

```text
C^T[N, M] = B^T[N, K] * A^T[K, M]
```

---

## Benchmark 结果

所有结果均在 NVIDIA A40 上测试。

### 方阵 GEMM

| Shape | naive | blocktile | thread2x2 | thread4x4 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| 256 x 256 x 256 | 1418.5 GFLOPS | 1743.0 GFLOPS | 3103.0 GFLOPS | 1759.8 GFLOPS | 4065.5 GFLOPS |
| 512 x 512 x 512 | 2038.1 GFLOPS | 2639.4 GFLOPS | 5456.8 GFLOPS | 7249.6 GFLOPS | 11212.3 GFLOPS |
| 1024 x 1024 x 1024 | 2261.1 GFLOPS | 2938.3 GFLOPS | 7531.3 GFLOPS | 10337.9 GFLOPS | 17346.2 GFLOPS |
| 2048 x 2048 x 2048 | 2115.2 GFLOPS | 2843.1 GFLOPS | 6610.5 GFLOPS | 11998.6 GFLOPS | 20045.9 GFLOPS |

以 `1024 x 1024 x 1024` 为例：

```text
naive      : 0.9498 ms,  2261.1 GFLOPS
blocktile  : 0.7308 ms,  2938.3 GFLOPS
thread2x2  : 0.2851 ms,  7531.3 GFLOPS
thread4x4  : 0.2077 ms, 10337.9 GFLOPS
cuBLAS     : 0.1238 ms, 17346.2 GFLOPS
```

可以看到：

```text
shared memory block tiling 相比 naive 有明显提升；
2x2 thread tile 进一步提高了寄存器级复用；
4x4 thread tile 在大矩阵上性能更好；
cuBLAS 仍然是性能最强的工业级 baseline。
```

---

## LLM-like GEMM Shape

下面测试了一些更接近大模型 Linear / MLP 的矩阵形状。

| Shape | naive | blocktile | thread2x2 | thread4x4 | cuBLAS |
|---|---:|---:|---:|---:|---:|
| M=1, N=4096, K=4096 | 150.4 GFLOPS | 114.7 GFLOPS | 147.8 GFLOPS | 111.5 GFLOPS | 272.0 GFLOPS |
| M=16, N=4096, K=4096 | 1402.4 GFLOPS | 1832.5 GFLOPS | 2347.9 GFLOPS | 1786.7 GFLOPS | 3887.6 GFLOPS |
| M=32, N=4096, K=4096 | 1678.3 GFLOPS | 2243.8 GFLOPS | 4596.6 GFLOPS | 3562.0 GFLOPS | 8231.9 GFLOPS |
| M=128, N=4096, K=4096 | 2056.8 GFLOPS | 2683.4 GFLOPS | 6360.8 GFLOPS | 9563.4 GFLOPS | 17511.3 GFLOPS |
| M=32, N=11008, K=4096 | 2000.5 GFLOPS | 2599.9 GFLOPS | 5842.9 GFLOPS | 4406.1 GFLOPS | 6578.1 GFLOPS |
| M=32, N=4096, K=11008 | 1644.9 GFLOPS | 2229.5 GFLOPS | 4620.8 GFLOPS | 3567.3 GFLOPS | 9219.6 GFLOPS |

可以看到，4x4 thread tile 并不是在所有 shape 下都更好。

例如 `M=32, N=4096, K=4096`：

```text
thread2x2  : 4596.6 GFLOPS
thread4x4  : 3562.0 GFLOPS
```

原因是：

```text
2x2 kernel 的 block tile 是 32 x 32，比较匹配 M=32；
4x4 kernel 的 block tile 是 64 x 64，在 M=32 时会造成 M 方向资源浪费。
```

这说明：

```text
thread tile 并不是越大越好；
tile size 需要根据矩阵 shape、寄存器压力、occupancy 和计算密度综合选择。
```

---

## Nsight Compute 分析

对 `1024 x 1024 x 1024` 的 2x2 和 4x4 thread tile kernel 做了 Nsight Compute 分析。

| Metric | thread2x2 | thread4x4 |
|---|---:|---:|
| Duration | 379.39 us | 273.22 us |
| Memory Throughput | 82.04% | 64.03% |
| DRAM Throughput | 7.58% | 9.32% |
| L1/TEX Cache Throughput | 87.54% | 81.86% |
| L2 Cache Throughput | 37.37% | 26.75% |
| Registers / Thread | 40 | 60 |
| Static Shared Memory / Block | 4.10 KB | 8.19 KB |
| Theoretical Occupancy | 100% | 66.67% |
| Achieved Occupancy | 83.38% | 41.11% |

分析结论：

```text
4x4 kernel 的寄存器使用更多，achieved occupancy 更低；
但 4x4 每个线程计算 16 个输出元素，比 2x2 的 4 个输出元素有更强的寄存器级复用；
因此 4x4 在大矩阵上虽然 occupancy 更低，但整体性能更好。
```

这也说明：

```text
occupancy 不是越高越好；
当 occupancy 足够隐藏延迟后，更高的计算密度和寄存器复用可能更重要。
```

---

## 构建方式

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

---

## 运行 Benchmark

```bash
./build/gemm_lab
```

保存输出：

```bash
./build/gemm_lab | tee benchmark.txt
```

---

## 项目结构

```text
cuda-gemm-lab/
├── CMakeLists.txt
├── README.md
└── src/
    └── main.cu
```

---

## 主要收获

1. Naive GEMM 由于 global memory 重复读取较多，性能较低。
2. Shared memory block tiling 可以减少 global memory 重复访问。
3. Thread tile / register blocking 可以提高寄存器级数据复用和计算密度。
4. 4x4 thread tile 在大方阵 GEMM 上更快，但在小 M shape 下可能退化。
5. cuBLAS 作为工业级 baseline，性能仍然更高，可能受益于 Tensor Core / TF32、warp-level tiling、double buffering 和更成熟的调度优化。
6. GEMM kernel 优化高度依赖 shape，不存在一个对所有场景都最优的 tile size。

---

## 后续计划

- WMMA / Tensor Core GEMM
- RMSNorm CUDA kernel
- Softmax CUDA kernel
- RoPE CUDA kernel
- Linear + activation fusion
- Triton 版本的常见 LLM kernel

---

## WMMA / Tensor Core GEMM

在完成 FP32 CUDA core GEMM 优化后，本项目进一步实现了一个最小 WMMA GEMM kernel，用于理解 Tensor Core 的基本使用方式。

该版本使用：

```text
A: FP16, row-major
B: FP16, row-major
C: FP32, row-major
Accumulation: FP32
```

核心思想是：

```text
一个 warp 计算一个 16 x 16 的 C tile
K 方向每次推进 16
通过 wmma::mma_sync 调用 Tensor Core 完成矩阵乘加
```

对应的计算形式为：

```text
C[16, 16] += A[16, 16] x B[16, 16]
```

### WMMA 核心 API

最小 WMMA kernel 使用了以下 API：

| API | 作用 |
|---|---|
| `wmma::fragment` | 定义 warp 级矩阵 tile |
| `wmma::fill_fragment` | 初始化 accumulator fragment |
| `wmma::load_matrix_sync` | 从 global memory 加载 A/B tile |
| `wmma::mma_sync` | 调用 Tensor Core 执行矩阵乘加 |
| `wmma::store_matrix_sync` | 将 accumulator fragment 写回 global memory |

其中：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

表示一次执行：

```text
C[16,16] += A[16,16] x B[16,16]
```

### WMMA Benchmark

测试平台：NVIDIA A40。

| Shape | WMMA minimal | cuBLAS HGEMM |
|---|---:|---:|
| 256 x 256 x 256 | 6023.5 GFLOPS | 4222.7 GFLOPS |
| 512 x 512 x 512 | 14063.5 GFLOPS | 22103.2 GFLOPS |
| 1024 x 1024 x 1024 | 14802.0 GFLOPS | 82305.8 GFLOPS |
| 2048 x 2048 x 2048 | 16684.8 GFLOPS | 77293.0 GFLOPS |

以 `1024 x 1024 x 1024` 为例：

```text
wmma minimal : 0.1451 ms, 14802.0 GFLOPS
cuBLAS hgemm : 0.0261 ms, 82305.8 GFLOPS
```

### 分析

最小 WMMA kernel 已经能够调用 Tensor Core，因此相比手写 FP32 CUDA core GEMM 有一定提升。例如在 `1024 x 1024 x 1024` 上，WMMA minimal 达到约 `14.8 TFLOPS`，高于 FP32 `thread4x4` 版本的约 `10.3 TFLOPS`。

但它和 cuBLAS HGEMM 仍有明显差距，主要原因是当前版本仍然非常基础：

```text
一个 block 只有一个 warp
一个 warp 只计算一个 16 x 16 C tile
没有 block-level tiling
没有 shared memory staging
没有多 warp 协作
没有 double buffering
没有复杂的 memory layout 优化
```

因此，当前 WMMA 版本的意义主要是验证 Tensor Core / WMMA API 的基本使用方式，而不是追求最终性能。

下一步优化方向：

```text
一个 block 使用多个 warp
一个 block 计算更大的 C tile
使用 shared memory staging 缓存 A/B tile
优化 global memory coalescing
引入 double buffering / pipeline
继续对比 cuBLAS Tensor Core baseline
```
