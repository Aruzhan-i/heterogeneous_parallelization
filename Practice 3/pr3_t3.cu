#include <cuda_runtime.h> // Подключение основного API CUDA для работы с видеокартой
#include <device_launch_parameters.h> // Подключение параметров запуска ядер (threadIdx, blockIdx и др.)

#include <iostream> // Подключение библиотеки для ввода-вывода в консоль
#include <vector> // Подключение контейнера динамический массив (вектор)
#include <algorithm> // Подключение стандартных алгоритмов (используется для std::sort на CPU)
#include <random> // Подключение средств для генерации случайных чисел

// Макрос для автоматической проверки ошибок CUDA после выполнения функций API
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
// Функция восстановления свойств кучи (выполняется на стороне GPU)
// ------------------------------------------------------------
__device__ void heapify(int* data, int n, int i) {
    while (true) { // Бесконечный цикл для спуска элемента вниз по дереву
        int largest = i; // Инициализируем самый большой элемент как корень текущего поддерева
        int l = 2 * i + 1; // Вычисляем индекс левого дочернего элемента
        int r = 2 * i + 2; // Вычисляем индекс правого дочернего элемента

        // Если левый дочерний элемент больше текущего самого большого
        if (l < n && data[l] > data[largest]) largest = l;
        // Если правый дочерний элемент больше текущего самого большого
        if (r < n && data[r] > data[largest]) largest = r;

        // Если самый большой элемент все еще корень, значит куча в порядке — выходим
        if (largest == i) break;

        // Меняем местами корень и найденный наибольший элемент
        int tmp = data[i];
        data[i] = data[largest];
        data[largest] = tmp;

        // Переходим к следующему уровню поддерева
        i = largest;
    }
}

// ------------------------------------------------------------
// Ядро CUDA: выполняет ПОЛНУЮ пирамидальную сортировку (Heap Sort)
// ------------------------------------------------------------
__global__ void heapSortKernel(int* data, int n) {
    // Ограничиваем выполнение только одним потоком (первым потоком первого блока)
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    // 1. Построение кучи (перегруппировка массива)
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(data, n, i); // Вызываем heapify для каждого внутреннего узла
    }

    // 2. Один за другим извлекаем элементы из кучи
    for (int i = n - 1; i > 0; i--) {
        // Перемещаем текущий корень (максимум) в конец массива
        int tmp = data[0];
        data[0] = data[i];
        data[i] = tmp;
        
        // Вызываем heapify на уменьшенной куче, чтобы восстановить порядок
        heapify(data, i, 0);
    }
}

// ------------------------------------------------------------
// Функция-обертка для запуска Heap Sort на GPU из кода хоста
// ------------------------------------------------------------
void gpuHeapSort(int* d_data, int n) {
    heapSortKernel<<<1, 1>>>(d_data, n); // Запуск ядра: 1 блок, 1 поток
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения работы GPU
}

// ------------------------------------------------------------
// Проверка наличия и характеристик CUDA-устройства
// ------------------------------------------------------------
void checkCudaDevice() {
    int deviceCount = 0; // Переменная для количества найденных GPU
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount)); // Получаем количество устройств

    if (deviceCount == 0) { // Если видеокарт с поддержкой CUDA не найдено
        std::cerr << "No CUDA devices found. Exiting.\n";
        exit(EXIT_FAILURE);
    }

    int device = 0; // Используем первое доступное устройство
    cudaDeviceProp prop; // Структура для характеристик GPU
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device)); // Заполняем структуру данными
    CUDA_CHECK(cudaSetDevice(device)); // Делаем это устройство активным

    std::cout << "CUDA device detected:\n"; // Вывод названия видеокарты
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute capability: "
              << prop.major << "." << prop.minor << "\n"; // Версия архитектуры
    std::cout << "  Global memory: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n"; // Объем памяти в МБ
}

// ------------------------------------------------------------
// ГЛАВНАЯ ФУНКЦИЯ (Точка входа)
// ------------------------------------------------------------
int main() {
    checkCudaDevice(); // Вызов проверки оборудования

    const int N = 1'000'000; // Определяем размер массива (1 миллион элементов)

    std::vector<int> h_data(N); // Создаем массив на стороне CPU
    std::mt19937 rng(42); // Инициализируем генератор случайных чисел с зерном 42
    std::uniform_int_distribution<int> dist(0, 1'000'000); // Диапазон чисел

    for (int& x : h_data)
        x = dist(rng); // Заполняем массив случайными данными

    // Подготовка эталонного массива для сравнения
    std::vector<int> reference = h_data;
    std::sort(reference.begin(), reference.end()); // Сортируем его стандартными средствами CPU

    int* d_data; // Указатель на память внутри видеокарты
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int))); // Выделяем память на GPU
    CUDA_CHECK(cudaMemcpy(
        d_data,
        h_data.data(),
        N * sizeof(int),
        cudaMemcpyHostToDevice
    )); // Копируем данные с компьютера на видеокарту

    cudaEvent_t start, stop; // Создаем события CUDA для измерения времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Фиксируем время начала
    gpuHeapSort(d_data, N); // Запускаем сортировку на GPU
    cudaEventRecord(stop); // Фиксируем время окончания
    cudaEventSynchronize(stop); // Дожидаемся фактического завершения событий

    float ms = 0.0f; // Переменная для результата замера в миллисекундах
    cudaEventElapsedTime(&ms, start, stop); // Вычисляем разницу во времени

    CUDA_CHECK(cudaMemcpy(
        h_data.data(),
        d_data,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    )); // Копируем отсортированные данные обратно на CPU

    // --------------------------------------------------------
    // Проверка корректности результата
    // --------------------------------------------------------
    bool isCorrect = (h_data == reference); // Сравниваем два вектора

    std::cout << "GPU Heap Sort time: " << ms << " ms\n"; // Вывод времени работы
    if (isCorrect) {
        std::cout << "Result verification: PASSED (GPU result matches CPU sort)\n";
    } else {
        std::cout << "Result verification: FAILED\n"; // В случае ошибки выводим первое несовпадение
        for (int i = 0; i < 10; i++) {
            if (h_data[i] != reference[i]) {
                std::cout << "First mismatch at index " << i
                          << ": GPU=" << h_data[i]
                          << ", CPU=" << reference[i] << "\n";
                break;
            }
        }
    }

    cudaFree(d_data); // Освобождаем память на GPU
    cudaEventDestroy(start); // Удаляем события
    cudaEventDestroy(stop);

    return 0; // Конец программы
}