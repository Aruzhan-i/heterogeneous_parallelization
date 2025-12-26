#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ------------------------------------------------------------
// Device heapify (sequential)
// ------------------------------------------------------------
__device__ void heapify(int* data, int n, int i) {
    while (true) {
        int largest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;

        if (l < n && data[l] > data[largest]) largest = l;
        if (r < n && data[r] > data[largest]) largest = r;

        if (largest == i) break;

        int tmp = data[i];
        data[i] = data[largest];
        data[largest] = tmp;

        i = largest;
    }
}

// ------------------------------------------------------------
// Kernel: FULL heap sort in one block (correctness-focused)
// ------------------------------------------------------------
__global__ void heapSortKernel(int* data, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    // 1. Build heap
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(data, n, i);
    }

    // 2. Extract elements
    for (int i = n - 1; i > 0; i--) {
        int tmp = data[0];
        data[0] = data[i];
        data[i] = tmp;
        heapify(data, i, 0);
    }
}

// ------------------------------------------------------------
// GPU Heap Sort wrapper
// ------------------------------------------------------------
void gpuHeapSort(int* d_data, int n) {
    heapSortKernel<<<1, 1>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------
// CUDA availability check
// ------------------------------------------------------------
void checkCudaDevice() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting.\n";
        exit(EXIT_FAILURE);
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
// MAIN
// ------------------------------------------------------------
int main() {
    checkCudaDevice();

    const int N = 1'000'000;

    std::vector<int> h_data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int& x : h_data)
        x = dist(rng);

    // Reference CPU sort
    std::vector<int> reference = h_data;
    std::sort(reference.begin(), reference.end());

    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        h_data.data(),
        N * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuHeapSort(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(
        h_data.data(),
        d_data,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    // --------------------------------------------------------
    // Correctness check (clear output)
    // --------------------------------------------------------
    bool isCorrect = (h_data == reference);

    std::cout << "GPU Heap Sort time: " << ms << " ms\n";
    if (isCorrect) {
        std::cout << "Result verification: PASSED (GPU result matches CPU sort)\n";
    } else {
        std::cout << "Result verification: FAILED\n";
        for (int i = 0; i < 10; i++) {
            if (h_data[i] != reference[i]) {
                std::cout << "First mismatch at index " << i
                          << ": GPU=" << h_data[i]
                          << ", CPU=" << reference[i] << "\n";
                break;
            }
        }
    }

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
