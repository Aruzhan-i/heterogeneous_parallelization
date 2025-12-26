#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ============================================================
   ===================== CPU SORTS ============================
   ============================================================ */

// -------- CPU MERGE SORT --------
void merge(std::vector<int>& a, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    std::vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = a[l + i];
    for (int j = 0; j < n2; j++) R[j] = a[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        a[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
}

void cpuMergeSort(std::vector<int>& a, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    cpuMergeSort(a, l, m);
    cpuMergeSort(a, m + 1, r);
    merge(a, l, m, r);
}

// -------- CPU QUICK SORT --------
void cpuQuickSort(std::vector<int>& a, int l, int r) {
    if (l >= r) return;
    int pivot = a[(l + r) / 2];
    int i = l, j = r;

    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
            std::swap(a[i], a[j]);
            i++; j--;
        }
    }
    cpuQuickSort(a, l, j);
    cpuQuickSort(a, i, r);
}

// -------- CPU HEAP SORT --------
void heapify(std::vector<int>& a, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && a[l] > a[largest]) largest = l;
    if (r < n && a[r] > a[largest]) largest = r;

    if (largest != i) {
        std::swap(a[i], a[largest]);
        heapify(a, n, largest);
    }
}

void cpuHeapSort(std::vector<int>& a) {
    int n = a.size();
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(a, n, i);

    for (int i = n - 1; i > 0; i--) {
        std::swap(a[0], a[i]);
        heapify(a, i, 0);
    }
}

/* ============================================================
   ===================== GPU SORTS ============================
   ============================================================ */

// -------- GPU MERGE SORT --------
__device__ void d_merge(const int* src, int* dst, int l, int m, int r) {
    int i = l, j = m, k = l;
    while (i < m && j < r)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < m) dst[k++] = src[i++];
    while (j < r) dst[k++] = src[j++];
}

__global__ void mergePass(const int* src, int* dst, int n, int width) {
    int seg = blockIdx.x;
    int l = seg * (2 * width);
    int m = min(l + width, n);
    int r = min(l + 2 * width, n);

    if (l < n && threadIdx.x == 0)
        d_merge(src, dst, l, m, r);
}

float gpuMergeSort(std::vector<int>& a) {
    int n = a.size();
    int *d_a, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);

    int* src = d_a;
    int* dst = d_tmp;

    for (int w = 1; w < n; w *= 2) {
        int segs = (n + 2 * w - 1) / (2 * w);
        mergePass<<<segs, 1>>>(src, dst, n, w);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(src, dst);
    }

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(a.data(), src, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_a);
    cudaFree(d_tmp);

    return ms;
}

// -------- GPU QUICK SORT --------
__device__ void d_quick(int* a, int l, int r) {
    if (l >= r) return;
    int p = a[(l + r) / 2];
    int i = l, j = r;

    while (i <= j) {
        while (a[i] < p) i++;
        while (a[j] > p) j--;
        if (i <= j) {
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
            i++; j--;
        }
    }
    d_quick(a, l, j);
    d_quick(a, i, r);
}

__global__ void quickKernel(int* a, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        d_quick(a, 0, n - 1);
}

float gpuQuickSort(std::vector<int>& a) {
    int n = a.size();
    int* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);

    quickKernel<<<1, 1>>>(d, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d);

    return ms;
}

// -------- GPU HEAP SORT --------
__device__ void d_heapify(int* a, int n, int i) {
    while (true) {
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int largest = i;

        if (l < n && a[l] > a[largest]) largest = l;
        if (r < n && a[r] > a[largest]) largest = r;
        if (largest == i) break;

        int t = a[i];
        a[i] = a[largest];
        a[largest] = t;
        i = largest;
    }
}

__global__ void heapKernel(int* a, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = n / 2 - 1; i >= 0; i--)
        d_heapify(a, n, i);

    for (int i = n - 1; i > 0; i--) {
        int t = a[0];
        a[0] = a[i];
        a[i] = t;
        d_heapify(a, i, 0);
    }
}

float gpuHeapSort(std::vector<int>& a) {
    int n = a.size();
    int* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);

    heapKernel<<<1, 1>>>(d, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d);

    return ms;
}

/* ============================================================
   ===================== BENCHMARK ============================
   ============================================================ */

template <typename Func>
double cpuTime(Func func, std::vector<int> a) {
    auto start = std::chrono::high_resolution_clock::now();
    func(a);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
}


int main() {
    checkCudaDevice();

    std::vector<int> sizes = {10000, 100000, 1000000};
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int N : sizes) {
        std::vector<int> base(N);
        for (int& x : base) x = dist(rng);

        std::vector<int> reference = base;
        std::sort(reference.begin(), reference.end());

        std::cout << "\nN = " << N << "\n";

        std::vector<int> a;

        // ---------------- CPU ----------------
        a = base;
        std::cout << "CPU Merge: "
                  << cpuTime([&](std::vector<int>& v){
                         cpuMergeSort(v, 0, v.size() - 1);
                     }, a)
                  << " ms\n";

        a = base;
        std::cout << "CPU Quick: "
                  << cpuTime([&](std::vector<int>& v){
                         cpuQuickSort(v, 0, v.size() - 1);
                     }, a)
                  << " ms\n";

        a = base;
        std::cout << "CPU Heap : "
                  << cpuTime(cpuHeapSort, a)
                  << " ms\n";

        // ---------------- GPU MERGE ----------------
        a = base;
        float tMerge = gpuMergeSort(a);
        std::cout << "GPU Merge: " << tMerge << " ms | "
                  << ((a == reference) ? "PASSED" : "FAILED") << "\n";

        // ---------------- GPU QUICK ----------------
        a = base;
        float tQuick = gpuQuickSort(a);
        std::cout << "GPU Quick: " << tQuick << " ms | "
                  << ((a == reference) ? "PASSED" : "FAILED") << "\n";

        // ---------------- GPU HEAP ----------------
        a = base;
        float tHeap = gpuHeapSort(a);
        std::cout << "GPU Heap : " << tHeap << " ms | "
                  << ((a == reference) ? "PASSED" : "FAILED") << "\n";
    }

    return 0;
}
