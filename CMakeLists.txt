cmake_minimum_required(VERSION 3.18)

project(cuda_gemm_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_lab
    src/main.cu
)

target_compile_options(gemm_lab PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

target_include_directories(gemm_lab PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemm_lab PRIVATE CUDA::cublas)