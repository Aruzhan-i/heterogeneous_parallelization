#include <cuda_runtime.h> // Подключение основного API CUDA для работы с видеокартой
#include <device_launch_parameters.h> // Подключение параметров запуска ядер (gridDim, blockDim и т.д.)

#include <iostream> // Подключение стандартной библиотеки ввода-вывода
#include <vector> // Подключение контейнера динамический массив (vector)
#include <algorithm> // Подключение стандартных алгоритмов (например, std::sort)
#include <random> // Подключение генератора случайных чисел
#include <climits> // Подключение констант предельных значений (например, INT_MAX)

// Макрос для проверки ошибок CUDA после выполнения функций API
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
// Функция для проверки наличия и свойств графического процессора (GPU)
// ------------------------------------------------------------
void checkCudaDevice() {
    int deviceCount = 0; // Переменная для хранения количества найденных GPU
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount)); // Получение количества доступных CUDA-устройств

    if (deviceCount == 0) { // Если устройств не найдено
        std::cerr << "No CUDA devices found. Exiting.\n"; // Вывод сообщения об ошибке
        std::exit(EXIT_FAILURE); // Завершение программы с кодом ошибки
    }

    int device = 0; // Идентификатор используемого устройства (по умолчанию 0)
    cudaDeviceProp prop; // Структура для хранения характеристик видеокарты
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device)); // Заполнение структуры характеристиками GPU
    CUDA_CHECK(cudaSetDevice(device)); // Установка выбранного устройства как активного

    std::cout << "CUDA device detected:\n"; // Вывод заголовка информации об устройстве
    std::cout << "  Name: " << prop.name << "\n"; // Вывод названия видеокарты
    std::cout << "  Compute capability: "
              << prop.major << "." << prop.minor << "\n"; // Вывод версии вычислительных возможностей (архитектуры)
    std::cout << "  Global memory: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n"; // Вывод общего объема видеопамяти в мегабайтах
}

// ------------------------------------------------------------
// 1. Потоковая функция (ядро) для сортировки внутри одного блока (Bitonic Sort)
// ------------------------------------------------------------
__global__ void blockSort(int* data, int n) {
    extern __shared__ int s[]; // Объявление динамической разделяемой памяти (shared memory) внутри блока

    int tid = threadIdx.x; // Получение локального индекса потока внутри блока
    int gid = blockIdx.x * blockDim.x + tid; // Вычисление глобального индекса элемента в массиве

    s[tid] = (gid < n) ? data[gid] : INT_MAX; // Копирование данных из глобальной памяти в разделяемую (или INT_MAX для выравнивания)
    __syncthreads(); // Барьерная синхронизация: ждем, пока все потоки заполнят разделяемую память

    // Внешний цикл битонной сортировки: определяет длину текущей битонной последовательности (k)
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        // Внутренний цикл: определяет шаг сравнения (j) внутри последовательности
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j; // Вычисление индекса элемента для сравнения с текущим с помощью XOR
            if (ixj > tid) { // Условие, чтобы каждая пара сравнивалась только один раз
                bool asc = ((tid & k) == 0); // Определение направления сортировки (по возрастанию или убыванию)
                int a = s[tid]; // Извлечение значения первого элемента пары
                int b = s[ixj]; // Извлечение значения второго элемента пары
                if ((asc && a > b) || (!asc && a < b)) { // Проверка необходимости обмена значений
                    s[tid] = b; // Запись меньшего/большего значения в текущий поток
                    s[ixj] = a; // Запись оставшегося значения во второй поток пары
                }
            }
            __syncthreads(); // Синхронизация потоков после каждого шага сравнения
        }
    }

    if (gid < n) data[gid] = s[tid]; // Запись отсортированных данных из разделяемой памяти обратно в глобальную
}

// ------------------------------------------------------------
// 2. Потоковая функция (ядро) для параллельного слияния отсортированных участков
// ------------------------------------------------------------
__global__ void mergeKernel(const int* src, int* dst, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального индекса потока
    int start = tid * 2 * width; // Вычисление начальной позиции для слияния двух подмассивов

    if (start >= n) return; // Завершение, если начальная позиция за пределами массива

    int mid = min(start + width, n); // Вычисление середины (границы между двумя подмассивами)
    int end = min(start + 2 * width, n); // Вычисление конца второго подмассива

    int i = start, j = mid, k = start; // Инициализация указателей для слияния (i - левый, j - правый, k - результат)

    // Стандартный алгоритм слияния: выбираем наименьший элемент из двух частей
    while (i < mid && j < end)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    // Копирование оставшихся элементов из левой части (если есть)
    while (i < mid) dst[k++] = src[i++];
    // Копирование оставшихся элементов из правой части (если есть)
    while (j < end) dst[k++] = src[j++];
}

// ------------------------------------------------------------
// 3. Основная управляющая функция сортировки на GPU
// ------------------------------------------------------------
void gpuMergeSort(int* d_data, int n) {
    int* d_tmp = nullptr; // Указатель на временный массив в видеопамяти
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(int))); // Выделение памяти под временный массив

    const int BLOCK = 1024; // Размер блока потоков для начальной сортировки
    const int MERGE_THREADS = 256; // Количество потоков в блоке для ядер слияния

    int gridSort = (n + BLOCK - 1) / BLOCK; // Расчет количества блоков для покрытия всего массива

    // Запуск начальной сортировки блоков данных в разделяемой памяти
    blockSort<<<gridSort, BLOCK, BLOCK * sizeof(int)>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения работы всех блоков

    int* src = d_data; // Исходный указатель на данные для итерации слияния
    int* dst = d_tmp; // Целевой указатель на данные для итерации слияния

    // Цикл слияния: на каждом шаге ширина сливаемых участков удваивается
    for (int width = BLOCK; width < n; width <<= 1) {
        int pairs = (n + 2 * width - 1) / (2 * width); // Вычисление общего количества пар для слияния
        int blocks = (pairs + MERGE_THREADS - 1) / MERGE_THREADS; // Вычисление количества блоков для слияния

        // Запуск ядра слияния
        mergeKernel<<<blocks, MERGE_THREADS>>>(src, dst, n, width);
        CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения слияния

        std::swap(src, dst); // Смена ролей массивов (результат становится источником для следующего шага)
    }

    // Если финальный результат оказался во временном массиве, копируем его обратно
    if (src != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, src, n * sizeof(int),
                              cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_tmp)); // Освобождение временной видеопамяти
}

// ------------------------------------------------------------
// 4. Главная функция программы
// ------------------------------------------------------------
int main() {
    checkCudaDevice(); // Проверка доступности GPU

    const int N = 1'000'000; // Количество элементов для сортировки (1 миллион)

    std::vector<int> h_data(N); // Создание вектора в оперативной памяти (хост)
    std::mt19937 rng(42); // Инициализация генератора Вихрь Мерсенна зерном 42
    std::uniform_int_distribution<int> dist(0, 1'000'000); // Диапазон случайных чисел

    // Заполнение массива случайными числами
    for (int& x : h_data) x = dist(rng);

    std::vector<int> reference = h_data; // Создание копии массива для эталонной сортировки
    std::sort(reference.begin(), reference.end()); // Сортировка эталона средствами CPU (стандартная библиотека)

    int* d_data = nullptr; // Указатель для данных на видеокарте (девайс)
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int))); // Выделение памяти на видеокарте
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                          N * sizeof(int),
                          cudaMemcpyHostToDevice)); // Копирование данных с компьютера на видеокарту

    cudaEvent_t start, stop; // Создание событий CUDA для измерения времени
    CUDA_CHECK(cudaEventCreate(&start)); // Инициализация начального события
    CUDA_CHECK(cudaEventCreate(&stop)); // Инициализация конечного события

    CUDA_CHECK(cudaEventRecord(start)); // Фиксация времени начала сортировки
    gpuMergeSort(d_data, N); // Вызов основной функции сортировки на GPU
    CUDA_CHECK(cudaEventRecord(stop)); // Фиксация времени окончания сортировки
    CUDA_CHECK(cudaEventSynchronize(stop)); // Ожидание завершения всех операций на GPU до этого момента

    float ms = 0.0f; // Переменная для хранения времени выполнения в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Расчет разницы во времени между событиями

    // Копирование отсортированных данных обратно с видеокарты в оперативную память
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                          N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool isCorrect = (h_data == reference); // Сравнение результата GPU с эталонным результатом CPU

    std::cout << "GPU Merge Sort time: " << ms << " ms\n"; // Вывод времени работы
    if (isCorrect) {
        std::cout << "Result verification: PASSED (GPU result matches CPU sort)\n"; // Сообщение при успехе
    } else {
        std::cout << "Result verification: FAILED\n"; // Сообщение при ошибке
    }

    CUDA_CHECK(cudaFree(d_data)); // Освобождение выделенной памяти на видеокарте
    CUDA_CHECK(cudaEventDestroy(start)); // Удаление объекта начального события
    CUDA_CHECK(cudaEventDestroy(stop)); // Удаление объекта конечного события

    return 0; // Завершение программы
}