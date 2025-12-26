#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <climits>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// ------------------------------------------------------------
// CUDA availability check
// ------------------------------------------------------------
void checkCudaDevice() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting.\n";
        std::exit(EXIT_FAILURE);
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaSetDevice(device));

    std::cout << "CUDA device detected:\n";
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "  Global memory: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";
}

// ------------------------------------------------------------
// 1. Block-level Bitonic Sort (shared memory)
// ------------------------------------------------------------
__global__ void blockSort(int* data, int n) {
    extern __shared__ int s[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    s[tid] = (gid < n) ? data[gid] : INT_MAX;
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool asc = ((tid & k) == 0);
                int a = s[tid];
                int b = s[ixj];
                if ((asc && a > b) || (!asc && a < b)) {
                    s[tid] = b;
                    s[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (gid < n) data[gid] = s[tid];
}

// ------------------------------------------------------------
// 2. Parallel merge kernel
// ------------------------------------------------------------
__global__ void mergeKernel(const int* src, int* dst, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * 2 * width;

    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start, j = mid, k = start;

    while (i < mid && j < end)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid) dst[k++] = src[i++];
    while (j < end) dst[k++] = src[j++];
}

// ------------------------------------------------------------
// 3. GPU Merge Sort
// ------------------------------------------------------------
void gpuMergeSort(int* d_data, int n) {
    int* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(int)));

    const int BLOCK = 1024;
    const int MERGE_THREADS = 256;

    int gridSort = (n + BLOCK - 1) / BLOCK;

    blockSort<<<gridSort, BLOCK, BLOCK * sizeof(int)>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int* src = d_data;
    int* dst = d_tmp;

    for (int width = BLOCK; width < n; width <<= 1) {
        int pairs = (n + 2 * width - 1) / (2 * width);
        int blocks = (pairs + MERGE_THREADS - 1) / MERGE_THREADS;

        mergeKernel<<<blocks, MERGE_THREADS>>>(src, dst, n, width);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(src, dst);
    }

    if (src != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, src, n * sizeof(int),
                              cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_tmp));
}

// ------------------------------------------------------------
// 4. Main
// ------------------------------------------------------------
int main() {
    checkCudaDevice();

    const int N = 1'000'000;

    std::vector<int> h_data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int& x : h_data) x = dist(rng);

    std::vector<int> reference = h_data;
    std::sort(reference.begin(), reference.end());

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                          N * sizeof(int),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gpuMergeSort(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                          N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool isCorrect = (h_data == reference);

    std::cout << "GPU Merge Sort time: " << ms << " ms\n";
    if (isCorrect) {
        std::cout << "Result verification: PASSED (GPU result matches CPU sort)\n";
    } else {
        std::cout << "Result verification: FAILED\n";
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
