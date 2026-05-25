cmake_minimum_required(VERSION 3.18)

project(cuda_gemm_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Use system CUDA Toolkit
set(CUDA_ROOT /usr/local/cuda-13.1)
set(CUDA_TARGET ${CUDA_ROOT}/targets/x86_64-linux)

include_directories(${CUDA_TARGET}/include)
link_directories(${CUDA_TARGET}/lib)

add_executable(gemm_lab
    src/main.cu
)

add_executable(wmma_lab
    src/wmma_benchmark.cu
)

target_compile_options(gemm_lab PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

target_compile_options(wmma_lab PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

target_link_libraries(gemm_lab PRIVATE cublas)
target_link_libraries(wmma_lab PRIVATE cublas)

# runtime library path
set_target_properties(gemm_lab PROPERTIES
    BUILD_RPATH ${CUDA_TARGET}/lib
)

set_target_properties(wmma_lab PROPERTIES
    BUILD_RPATH ${CUDA_TARGET}/lib
)
