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
// Device quick sort (recursive, small ranges)
// ------------------------------------------------------------
__device__ void deviceQuickSort(int* a, int l, int r) {
    if (l >= r) return;

    int pivot = a[(l + r) / 2];
    int i = l, j = r;

    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
            i++; j--;
        }
    }

    if (l < j) deviceQuickSort(a, l, j);
    if (i < r) deviceQuickSort(a, i, r);
}

// ------------------------------------------------------------
// Kernel: final sort (single block, correctness-focused)
// ------------------------------------------------------------
__global__ void quickSortKernel(int* data, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        deviceQuickSort(data, 0, n - 1);
    }
}

// ------------------------------------------------------------
// GPU Quick Sort wrapper
// ------------------------------------------------------------
void gpuQuickSort(int* d_data, int n) {
    quickSortKernel<<<1, 1>>>(d_data, n);
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
    gpuQuickSort(d_data, N);
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

    std::cout << "GPU Quick Sort time: " << ms << " ms\n";
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
