cmake_minimum_required(VERSION 3.18)

project(heter LANGUAGES CXX CUDA)

# Host C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA MUST be C++14 for GCC 11 compatibility
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(pr3_t1 pr3_t1.cu)

set_target_properties(pr3_t1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


add_executable(pr3_t2 pr3_t2.cu)

set_target_properties(pr3_t2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
add_executable(pr3_t3 pr3_t3.cu)

set_target_properties(pr3_t3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)



add_executable(pr3_t4 pr3_t4.cu)
set_target_properties(pr3_t4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
