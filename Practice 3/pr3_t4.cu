#include <cuda_runtime.h> // Основной API CUDA для управления памятью и устройствами
#include <device_launch_parameters.h> // Параметры запуска ядер (threadIdx, blockIdx и т.д.)

#include <iostream> // Стандартный ввод-вывод
#include <vector> // Динамические массивы C++
#include <algorithm> // Стандартные алгоритмы (std::swap, std::sort)
#include <random> // Генерация случайных чисел
#include <chrono> // Высокоточное измерение времени на CPU

// Макрос для проверки ошибок CUDA API
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
   ===================== CPU SORTS (Сортировки на ЦП) =========
   ============================================================ */

// -------- CPU MERGE SORT (Сортировка слиянием на CPU) --------
void merge(std::vector<int>& a, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m; // Размеры двух подмассивов
    std::vector<int> L(n1), R(n2); // Временные массивы для левой и правой частей
    for (int i = 0; i < n1; i++) L[i] = a[l + i]; // Копирование левой части
    for (int j = 0; j < n2; j++) R[j] = a[m + 1 + j]; // Копирование правой части

    int i = 0, j = 0, k = l; // Индексы для слияния
    while (i < n1 && j < n2) // Сравниваем элементы и записываем меньший
        a[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) a[k++] = L[i++]; // Дописываем остатки левой части
    while (j < n2) a[k++] = R[j++]; // Дописываем остатки правой части
}

void cpuMergeSort(std::vector<int>& a, int l, int r) {
    if (l >= r) return; // Базовый случай рекурсии
    int m = (l + r) / 2; // Нахождение середины
    cpuMergeSort(a, l, m); // Рекурсивная сортировка левой половины
    cpuMergeSort(a, m + 1, r); // Рекурсивная сортировка правой половины
    merge(a, l, m, r); // Слияние двух отсортированных половин
}

// -------- CPU QUICK SORT (Быстрая сортировка на CPU) --------
void cpuQuickSort(std::vector<int>& a, int l, int r) {
    if (l >= r) return; // Выход из рекурсии
    int pivot = a[(l + r) / 2]; // Выбор опорного элемента
    int i = l, j = r;

    while (i <= j) { // Разбиение массива
        while (a[i] < pivot) i++; // Поиск элемента больше опорного слева
        while (a[j] > pivot) j--; // Поиск элемента меньше опорного справа
        if (i <= j) {
            std::swap(a[i], a[j]); // Обмен элементов
            i++; j--;
        }
    }
    cpuQuickSort(a, l, j); // Рекурсия для левой части
    cpuQuickSort(a, i, r); // Рекурсия для правой части
}

// -------- CPU HEAP SORT (Пирамидальная сортировка на CPU) --------
void heapify(std::vector<int>& a, int n, int i) {
    int largest = i; // Инициализация наибольшего элемента как корня
    int l = 2 * i + 1; // Левый потомок
    int r = 2 * i + 2; // Правый потомок

    if (l < n && a[l] > a[largest]) largest = l; // Проверка левого потомка
    if (r < n && a[r] > a[largest]) largest = r; // Проверка правого потомка

    if (largest != i) {
        std::swap(a[i], a[largest]); // Если корень не самый большой, меняем
        heapify(a, n, largest); // Рекурсивно восстанавливаем кучу ниже
    }
}

void cpuHeapSort(std::vector<int>& a) {
    int n = a.size();
    for (int i = n / 2 - 1; i >= 0; i--) // Построение кучи (max-heap)
        heapify(a, n, i);

    for (int i = n - 1; i > 0; i--) {
        std::swap(a[0], a[i]); // Перенос максимума в конец
        heapify(a, i, 0); // Восстановление кучи для оставшихся элементов
    }
}

/* ============================================================
   ===================== GPU SORTS (Сортировки на ГП) =========
   ============================================================ */

// -------- GPU MERGE SORT (Слияние на GPU) --------
__device__ void d_merge(const int* src, int* dst, int l, int m, int r) {
    int i = l, j = m, k = l; // Указатели для слияния двух сегментов
    while (i < m && j < r)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < m) dst[k++] = src[i++]; // Копирование остатков левой части
    while (j < r) dst[k++] = src[j++]; // Копирование остатков правой части
}

__global__ void mergePass(const int* src, int* dst, int n, int width) {
    int seg = blockIdx.x; // Каждый блок обрабатывает одну пару сегментов
    int l = seg * (2 * width); // Левая граница
    int m = min(l + width, n); // Середина (граница между сегментами)
    int r = min(l + 2 * width, n); // Правая граница

    if (l < n && threadIdx.x == 0) // Только первый поток блока выполняет слияние
        d_merge(src, dst, l, m, r);
}

float gpuMergeSort(std::vector<int>& a) {
    int n = a.size();
    int *d_a, *d_tmp; // Указатели на память ГП
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int))); // Выделение основной памяти
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(int))); // Выделение временной памяти
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice)); // Копирование данных на ГП

    cudaEvent_t s, e; // События для замера времени
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s); // Старт таймера

    int* src = d_a; // Текущий источник данных
    int* dst = d_tmp; // Куда записывать результат

    for (int w = 1; w < n; w *= 2) { // Итеративное удвоение ширины сегментов
        int segs = (n + 2 * w - 1) / (2 * w); // Количество пар сегментов
        mergePass<<<segs, 1>>>(src, dst, n, w); // Запуск ядра (один поток на пару)
        CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения прохода
        std::swap(src, dst); // Меняем местами источник и приемник
    }

    cudaEventRecord(e); // Остановка таймера
    cudaEventSynchronize(e);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e); // Расчет времени выполнения

    CUDA_CHECK(cudaMemcpy(a.data(), src, n * sizeof(int), cudaMemcpyDeviceToHost)); // Возврат данных на хост
    cudaFree(d_a); // Освобождение памяти
    cudaFree(d_tmp);

    return ms;
}

// -------- GPU QUICK SORT (Быстрая на GPU - последовательная) --------
__device__ void d_quick(int* a, int l, int r) {
    if (l >= r) return; // Выход из рекурсии
    int p = a[(l + r) / 2]; // Опорный элемент
    int i = l, j = r;

    while (i <= j) { // Алгоритм разделения Хоара на ГП
        while (a[i] < p) i++;
        while (a[j] > p) j--;
        if (i <= j) {
            int t = a[i]; a[i] = a[j]; a[j] = t; // Обмен
            i++; j--;
        }
    }
    d_quick(a, l, j); // Рекурсивный вызов (требует Compute Capability 3.5+)
    d_quick(a, i, r);
}

__global__ void quickKernel(int* a, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) // Выполняется только одним потоком
        d_quick(a, 0, n - 1);
}

float gpuQuickSort(std::vector<int>& a) {
    int n = a.size();
    int* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int))); // Выделение памяти на ГП
    CUDA_CHECK(cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);

    quickKernel<<<1, 1>>>(d, n); // Запуск одним потоком (медленно для больших N)
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d);

    return ms;
}

// -------- GPU HEAP SORT (Пирамидальная на GPU) --------
__device__ void d_heapify(int* a, int n, int i) {
    while (true) { // Итеративное восстановление свойств кучи
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int largest = i;

        if (l < n && a[l] > a[largest]) largest = l;
        if (r < n && a[r] > a[largest]) largest = r;
        if (largest == i) break;

        int t = a[i]; a[i] = a[largest]; a[largest] = t; // Обмен с потомком
        i = largest; // Спуск вниз по дереву
    }
}

__global__ void heapKernel(int* a, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return; // Только один поток

    for (int i = n / 2 - 1; i >= 0; i--) // Этап 1: Построение кучи
        d_heapify(a, n, i);

    for (int i = n - 1; i > 0; i--) { // Этап 2: Извлечение элементов
        int t = a[0]; a[0] = a[i]; a[i] = t;
        d_heapify(a, i, 0); // Восстановление кучи
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

    heapKernel<<<1, 1>>>(d, n); // Последовательное выполнение на ГП
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
   ===================== BENCHMARK (Тестирование) =============
   ============================================================ */

template <typename Func>
double cpuTime(Func func, std::vector<int> a) {
    auto start = std::chrono::high_resolution_clock::now(); // Замер времени на CPU
    func(a);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void checkCudaDevice() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount)); // Поиск GPU

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device)); // Характеристики устройства
    CUDA_CHECK(cudaSetDevice(device));

    std::cout << "CUDA device detected:\n";
    std::cout << "  Name: " << prop.name << "\n"; // Имя видеокарты
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
}

int main() {
    checkCudaDevice(); // Инициализация ГП

    std::vector<int> sizes = {10000, 100000, 1000000}; // Наборы размеров данных
    std::mt19937 rng(42); // Генератор с фиксированным зерном
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int N : sizes) { // Тестирование для каждого размера
        std::vector<int> base(N);
        for (int& x : base) x = dist(rng); // Генерация случайных чисел

        std::vector<int> reference = base;
        std::sort(reference.begin(), reference.end()); // Эталонный результат (CPU std::sort)

        std::cout << "\nN = " << N << "\n";

        std::vector<int> a;

        // ---------------- CPU Сравнение ----------------
        a = base;
        std::cout << "CPU Merge: " << cpuTime([&](std::vector<int>& v){
                         cpuMergeSort(v, 0, v.size() - 1);
                     }, a) << " ms\n";

        a = base;
        std::cout << "CPU Quick: " << cpuTime([&](std::vector<int>& v){
                         cpuQuickSort(v, 0, v.size() - 1);
                     }, a) << " ms\n";

        a = base;
        std::cout << "CPU Heap : " << cpuTime(cpuHeapSort, a) << " ms\n";

        // ---------------- GPU Сравнение ----------------
        a = base;
        float tMerge = gpuMergeSort(a); // GPU Merge (наиболее параллельный из трех)
        std::cout << "GPU Merge: " << tMerge << " ms | " << ((a == reference) ? "PASSED" : "FAILED") << "\n";

        a = base;
        float tQuick = gpuQuickSort(a); // GPU Quick (один поток + рекурсия)
        std::cout << "GPU Quick: " << tQuick << " ms | " << ((a == reference) ? "PASSED" : "FAILED") << "\n";

        a = base;
        float tHeap = gpuHeapSort(a); // GPU Heap (один поток + циклы)
        std::cout << "GPU Heap : " << tHeap << " ms | " << ((a == reference) ? "PASSED" : "FAILED") << "\n";
    }

    return 0; // Конец программы
}