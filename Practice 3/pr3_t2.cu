#include <cuda_runtime.h> // Подключение основного API CUDA для управления устройством и памятью
#include <device_launch_parameters.h> // Подключение параметров запуска ядер (threadIdx, blockIdx и т.д.)

#include <iostream> // Подключение библиотеки для стандартного ввода и вывода (cout, cerr)
#include <vector> // Подключение контейнера динамический массив из стандартной библиотеки
#include <algorithm> // Подключение стандартных алгоритмов (используется для std::sort на CPU)
#include <random> // Подключение средств генерации случайных чисел

// Макрос для автоматической проверки ошибок CUDA после вызова функций API
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
// Рекурсивная функция быстрой сортировки, выполняемая непосредственно на GPU
// ------------------------------------------------------------
__device__ void deviceQuickSort(int* a, int l, int r) {
    if (l >= r) return; // Базовый случай рекурсии: если диапазон пуст или из одного элемента, выходим

    int pivot = a[(l + r) / 2]; // Выбор опорного элемента (в данном случае — средний элемент)
    int i = l, j = r; // Инициализация указателей для разделения массива

    // Процесс разбиения (partitioning)
    while (i <= j) {
        while (a[i] < pivot) i++; // Поиск элемента слева, который должен стоять справа от опорного
        while (a[j] > pivot) j--; // Поиск элемента справа, который должен стоять слева от опорного
        if (i <= j) { // Если указатели не пересеклись, меняем элементы местами
            int tmp = a[i]; // Временное хранение значения
            a[i] = a[j]; // Перестановка левого элемента
            a[j] = tmp; // Перестановка правого элемента
            i++; j--; // Сдвиг указателей после обмена
        }
    }

    if (l < j) deviceQuickSort(a, l, j); // Рекурсивный вызов для левой части массива
    if (i < r) deviceQuickSort(a, i, r); // Рекурсивный вызов для правой части массива
}

// ------------------------------------------------------------
// CUDA-ядро: точка входа для запуска сортировки на видеокарте
// ------------------------------------------------------------
__global__ void quickSortKernel(int* data, int n) {
    // Выполняем сортировку только одним потоком (первым потоком первого блока)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        deviceQuickSort(data, 0, n - 1); // Запуск рекурсивного алгоритма на GPU
    }
}

// ------------------------------------------------------------
// Обертка для запуска GPU Quick Sort из основного кода (Host)
// ------------------------------------------------------------
void gpuQuickSort(int* d_data, int n) {
    quickSortKernel<<<1, 1>>>(d_data, n); // Запуск ядра с конфигурацией: 1 блок, 1 поток
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения выполнения ядра на видеокарте
}

// ------------------------------------------------------------
// Функция для проверки наличия и получения информации о CUDA-устройстве
// ------------------------------------------------------------
void checkCudaDevice() {
    int deviceCount = 0; // Переменная для хранения количества GPU
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount)); // Запрос количества доступных видеокарт

    if (deviceCount == 0) { // Если видеокарт с поддержкой CUDA нет
        std::cerr << "No CUDA devices found. Exiting.\n"; // Вывод сообщения об ошибке
        exit(EXIT_FAILURE); // Завершение программы
    }

    int device = 0; // Индекс устройства (используем первое доступное)
    cudaDeviceProp prop; // Структура для хранения свойств устройства
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device)); // Получение характеристик GPU
    CUDA_CHECK(cudaSetDevice(device)); // Установка выбранного GPU как активного

    std::cout << "CUDA device detected:\n"; // Вывод заголовка
    std::cout << "  Name: " << prop.name << "\n"; // Вывод названия видеокарты
    std::cout << "  Compute capability: "
              << prop.major << "." << prop.minor << "\n"; // Вывод версии вычислительной архитектуры
    std::cout << "  Global memory: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n"; // Вывод объема памяти в МБ
}

// ------------------------------------------------------------
// Главная функция программы
// ------------------------------------------------------------
int main() {
    checkCudaDevice(); // Проверка и инициализация GPU

    const int N = 1'000'000; // Определение размера массива (1 миллион элементов)

    std::vector<int> h_data(N); // Резервирование памяти под массив на хосте (CPU)
    std::mt19937 rng(42); // Инициализация генератора случайных чисел (зерно 42)
    std::uniform_int_distribution<int> dist(0, 1'000'000); // Определение диапазона чисел

    for (int& x : h_data)
        x = dist(rng); // Заполнение массива на CPU случайными числами

    // Подготовка эталонного массива для проверки корректности
    std::vector<int> reference = h_data; // Копирование данных в эталонный массив
    std::sort(reference.begin(), reference.end()); // Сортировка эталона силами CPU

    int* d_data; // Указатель на данные в памяти видеокарты
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int))); // Выделение памяти на GPU
    CUDA_CHECK(cudaMemcpy(
        d_data,
        h_data.data(),
        N * sizeof(int),
        cudaMemcpyHostToDevice
    )); // Копирование неотсортированных данных с CPU на GPU

    cudaEvent_t start, stop; // Создание объектов событий для замера времени
    cudaEventCreate(&start); // Инициализация события начала
    cudaEventCreate(&stop); // Инициализация события окончания

    cudaEventRecord(start); // Запись метки времени начала
    gpuQuickSort(d_data, N); // Выполнение сортировки на GPU
    cudaEventRecord(stop); // Запись метки времени окончания
    cudaEventSynchronize(stop); // Ожидание физического завершения всех операций

    float ms = 0.0f; // Переменная для хранения времени в миллисекундах
    cudaEventElapsedTime(&ms, start, stop); // Вычисление времени между событиями

    CUDA_CHECK(cudaMemcpy(
        h_data.data(),
        d_data,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    )); // Копирование результата сортировки обратно с GPU на CPU

    // --------------------------------------------------------
    // Проверка правильности результата
    // --------------------------------------------------------
    bool isCorrect = (h_data == reference); // Сравнение массива GPU с результатом std::sort

    std::cout << "GPU Quick Sort time: " << ms << " ms\n"; // Вывод времени работы
    if (isCorrect) {
        std::cout << "Result verification: PASSED (GPU result matches CPU sort)\n"; // Сообщение об успехе
    } else {
        std::cout << "Result verification: FAILED\n"; // Сообщение об ошибке
        for (int i = 0; i < 10; i++) { // Вывод первых 10 несовпадений для отладки
            if (h_data[i] != reference[i]) {
                std::cout << "First mismatch at index " << i
                          << ": GPU=" << h_data[i]
                          << ", CPU=" << reference[i] << "\n";
                break;
            }
        }
    }

    cudaFree(d_data); // Освобождение выделенной памяти на видеокарте
    cudaEventDestroy(start); // Уничтожение объекта события начала
    cudaEventDestroy(stop); // Уничтожение объекта события окончания

    return 0; // Успешное завершение программы
}