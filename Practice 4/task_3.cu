// task_3.cu                                                       // Имя файла
// Задание 3: Оптимизация сортировки на GPU (учебная реализация)  // Описание задания
// Требования задания:                                            // Блок требований
// 1) Сортировка пузырьком небольших подмассивов                  // Требование 1: локальная сортировка
// 2) Хранение общего массива в глобальной памяти                 // Требование 2: global memory
// 3) Слияние с использованием shared memory                      // Требование 3: shared memory
// Важно: когда run становится большим...                         // Важное примечание о лимите shared

#include <cuda_runtime.h>                                         // CUDA Runtime API
#include <iostream>                                               // Ввод/вывод
#include <vector>                                                 // std::vector
#include <random>                                                 // Генерация случайных чисел
#include <iomanip>                                                // Форматированный вывод
#include <cstdlib>                                                // std::exit
#include <cmath>                                                  // Математические функции

// Макрос проверки ошибок CUDA (ВАЖНО: после '\' нельзя писать комментарии!)
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "Ошибка CUDA: " << cudaGetErrorString(err)    \
                  << " (код " << (int)err << ") в "               \
                  << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// Таймер на GPU (cudaEvent)
struct GpuTimer {                         // Структура таймера GPU
    cudaEvent_t start{}, stop{};          // События начала и конца
    GpuTimer(){                           // Конструктор
        CUDA_CHECK(cudaEventCreate(&start)); // Создание start event
        CUDA_CHECK(cudaEventCreate(&stop));  // Создание stop event
    }
    ~GpuTimer(){                          // Деструктор
        cudaEventDestroy(start);          // Удаление start event
        cudaEventDestroy(stop);           // Удаление stop event
    }
    void tic(cudaStream_t s=0){           // Начало замера
        CUDA_CHECK(cudaEventRecord(start, s)); // Запись start event
    }
    float toc(cudaStream_t s=0){          // Конец замера
        CUDA_CHECK(cudaEventRecord(stop, s));        // Запись stop event
        CUDA_CHECK(cudaEventSynchronize(stop));      // Ожидание stop event
        float ms=0.f;                                // Переменная времени
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Подсчёт времени
        return ms;                                   // Возврат времени
    }
};

// ------------------------------                                 // Раздел комментариев
// Настройки "маленького подмассива"                               // Пояснение параметров
// K = сколько элементов сортирует один поток                      // Значение K
// tile = K * blockDim.x                                           // Размер тайла
// ------------------------------                                 // Раздел комментариев
constexpr int K = 8;                                              // Число элементов на поток
constexpr float INF = 1e30f;                                      // Значение "бесконечность"

// Локальная пузырьковая сортировка K элементов
__device__ __forceinline__ void bubble_sort_local(float v[K]) {   // Device-функция сортировки
    #pragma unroll                                                // Разворачиваем цикл
    for (int i = 0; i < K; i++) {                                 // Внешний цикл
        #pragma unroll                                            // Разворачиваем внутренний цикл
        for (int j = 0; j < K - 1 - i; j++) {                      // Внутренний цикл
            if (v[j] > v[j + 1]) {                                 // Если элементы не по порядку
                float t = v[j];                                   // Временная переменная
                v[j] = v[j + 1];                                  // Обмен значений
                v[j + 1] = t;                                     // Завершение обмена
            }
        }
    }
}

// -----------------------------------------------------------     // Раздел этапа 1
__global__ void sort_tile_local_then_merge_shared(float* a, int n) { // Ядро этапа 1
    extern __shared__ float sh[];                                  // Shared память
    int tid = threadIdx.x;                                         // Индекс потока
    int block = blockIdx.x;                                        // Индекс блока

    int tile = blockDim.x * K;                                     // Размер тайла

    float* s   = sh;                                               // Указатель на данные
    float* tmp = sh + tile;                                        // Указатель на буфер

    int tile_base = block * tile;                                  // База тайла

    float loc[K];                                                  // Локальный массив
    int base = tile_base + tid * K;                                // База данных потока

    #pragma unroll                                                // Разворачиваем цикл
    for (int i = 0; i < K; i++) {                                 // Чтение данных
        int idx = base + i;                                       // Глобальный индекс
        loc[i] = (idx < n) ? a[idx] : INF;                        // Загрузка или INF
    }
    bubble_sort_local(loc);                                       // Локальная сортировка

    #pragma unroll                                                // Разворачиваем цикл
    for (int i = 0; i < K; i++) {                                 // Запись в shared
        s[tid * K + i] = loc[i];                                  // Копирование
    }
    __syncthreads();                                              // Синхронизация

    for (int run = K; run < tile; run *= 2) {                     // Итеративный merge
        for (int start = tid * (2 * run); start < tile; start += blockDim.x * (2 * run)) {
            int mid = start + run;                                // Середина
            int end = min(start + 2 * run, tile);                // Конец

            int i = start;                                       // Индекс i
            int j = mid;                                         // Индекс j
            int t = start;                                       // Индекс tmp

            while (i < mid && j < end) {                          // Пока есть элементы
                tmp[t++] = (s[i] <= s[j]) ? s[i++] : s[j++];      // Merge шаг
            }
            while (i < mid) tmp[t++] = s[i++];                   // Остаток A
            while (j < end) tmp[t++] = s[j++];                   // Остаток B
        }

        __syncthreads();                                          // Синхронизация

        for (int idx = tid; idx < tile; idx += blockDim.x) {     // Копирование назад
            s[idx] = tmp[idx];                                   // tmp → s
        }
        __syncthreads();                                         // Синхронизация
    }

    for (int idx = tid; idx < tile; idx += blockDim.x) {         // Запись в global
        int g = tile_base + idx;                                 // Глобальный индекс
        if (g < n) a[g] = s[idx];                                // Запись результата
    }
}

// -----------------------------------------------------------     // Раздел merge shared
__global__ void merge_pass_shared(const float* in, float* out, int n, int run) { // Ядро merge shared
    extern __shared__ float s[];                                  // Shared память

    int pair = blockIdx.x;                                        // Индекс пары
    int base = pair * 2 * run;                                    // База пары

    float* A = s;                                                 // Первый массив
    float* B = s + run;                                           // Второй массив

    for (int i = threadIdx.x; i < run; i += blockDim.x) {         // Загрузка в shared
        int ia = base + i;                                        // Индекс A
        int ib = base + run + i;                                  // Индекс B
        A[i] = (ia < n) ? in[ia] : INF;                           // Загрузка A
        B[i] = (ib < n) ? in[ib] : INF;                           // Загрузка B
    }
    __syncthreads();                                              // Синхронизация

    if (threadIdx.x == 0) {                                       // Только поток 0
        int i = 0, j = 0;                                         // Индексы
        for (int k = 0; k < 2 * run; k++) {                       // Merge цикл
            float x = (A[i] <= B[j]) ? A[i++] : B[j++];           // Выбор меньшего
            int outIdx = base + k;                                // Индекс вывода
            if (outIdx < n) out[outIdx] = x;                     // Запись
        }
    }
}

// -----------------------------------------------------------     // Раздел merge global
__device__ __forceinline__ float getA(const float* in, int base, int i, int run, int n) { // Доступ к A
    int idx = base + i;                                           // Индекс
    return (i >= 0 && i < run && idx < n) ? in[idx] : INF;        // Проверка границ
}
__device__ __forceinline__ float getB(const float* in, int base, int i, int run, int n) { // Доступ к B
    int idx = base + run + i;                                     // Индекс
    return (i >= 0 && i < run && idx < n) ? in[idx] : INF;        // Проверка границ
}

__global__ void merge_pass_global(const float* in, float* out, int n, int run) { // Ядро merge global
    int pair = blockIdx.x;                                        // Индекс пары
    int base = pair * 2 * run;                                    // База пары

    for (int k = threadIdx.x; k < 2 * run; k += blockDim.x) {     // Каждая нить — несколько k
        int i_min = max(0, k - run);                              // Минимальный i
        int i_max = min(k, run);                                  // Максимальный i

        while (i_min < i_max) {                                   // Бинарный поиск
            int i = (i_min + i_max) / 2;                          // Средний i
            int j = k - i;                                        // Соответствующий j

            float A_i   = getA(in, base, i,     run, n);          // Значение A
            float B_jm1 = getB(in, base, j - 1, run, n);          // Значение B

            if (A_i < B_jm1) i_min = i + 1;                       // Сдвиг вправо
            else i_max = i;                                      // Сдвиг влево
        }

        int i = i_min;                                           // Найденный i
        int j = k - i;                                           // Найденный j

        float A_im1 = getA(in, base, i - 1, run, n);              // A[i-1]
        float B_jm1 = getB(in, base, j - 1, run, n);              // B[j-1]

        float x = (A_im1 > B_jm1) ? A_im1 : B_jm1;                // Выбор элемента

        int outIdx = base + k;                                   // Индекс вывода
        if (outIdx < n) out[outIdx] = x;                         // Запись результата
    }
}

// Проверка на CPU: отсортирован ли массив
static bool is_sorted_cpu(const std::vector<float>& v) {         // Проверка сортировки
    for (size_t i = 1; i < v.size(); i++)                        // Проход по массиву
        if (v[i-1] > v[i]) return false;                         // Если не по порядку
    return true;                                                 // Всё отсортировано
}

// Генерация случайных данных
static std::vector<float> generate_data(int n) {                // Генерация данных
    std::mt19937 rng(123);                                       // RNG
    std::uniform_real_distribution<float> dist(0.f, 1.f);       // Диапазон [0,1]
    std::vector<float> h(n);                                     // Вектор
    for (int i = 0; i < n; i++) h[i] = dist(rng);                // Заполнение
    return h;                                                    // Возврат
}

int main() {                                                     // Точка входа
    std::cout << "=============================================\n"; // Заголовок
    std::cout << "Задание 3: Сортировка (local bubble + shared merge)\n"; // Название
    std::cout << "=============================================\n\n"; // Разделитель

    std::vector<int> sizes = { 10'000, 100'000, 1'000'000 };     // Размеры массивов

    const int block = 256;                                       // Размер блока
    const int tile  = block * K;                                 // Размер тайла

    cudaDeviceProp prop{};                                       // Структура свойств GPU
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));               // Получение свойств
    size_t shared_limit = prop.sharedMemPerBlock;                // Лимит shared

    std::cout << "Информация об устройстве:\n";                  // Заголовок
    std::cout << "  GPU: " << prop.name << "\n";                 // Имя GPU
    std::cout << "  Лимит shared memory на блок: " << shared_limit << " байт\n\n"; // Лимит

    for (int n : sizes) {                                        // Для каждого размера
        std::cout << "---------------------------------------------\n"; // Разделитель
        std::cout << "Размер массива n = " << n << "\n";         // Печать n
        std::cout << "Параметры: block = " << block << ", K = " << K
                  << " => tile = " << tile << " элементов на блок\n"; // Параметры

        auto h = generate_data(n);                               // Генерация данных

        float *d_a=nullptr, *d_b=nullptr;                        // Указатели GPU
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));         // Выделение d_a
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));         // Выделение d_b
        CUDA_CHECK(cudaMemcpy(d_a, h.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // Копирование

        int grid_tiles = (n + tile - 1) / tile;                  // Число блоков

        size_t sh1 = (size_t)(2 * tile) * sizeof(float);         // Shared для этапа 1

        GpuTimer t;                                              // Таймер
        t.tic();                                                 // Старт таймера
        sort_tile_local_then_merge_shared<<<grid_tiles, block, sh1>>>(d_a, n); // Запуск ядра
        CUDA_CHECK(cudaGetLastError());                           // Проверка ошибки
        CUDA_CHECK(cudaDeviceSynchronize());                      // Синхронизация
        float ms_stage1 = t.toc();                                // Время этапа 1

        float ms_merge_total = 0.f;                               // Время merge
        int run = tile;                                          // Начальный run

        bool ping = true;                                        // Переключатель буферов

        while (run < n) {                                        // Пока не отсортировано всё
            int pairs = (n + 2 * run - 1) / (2 * run);            // Число пар

            size_t sh = (size_t)(2 * run) * sizeof(float);       // Shared для merge

            t.tic();                                             // Старт таймера

            if (sh <= shared_limit) {                            // Если хватает shared
                merge_pass_shared<<<pairs, 256, sh>>>(ping ? d_a : d_b, ping ? d_b : d_a, n, run); // Shared merge
            } else {                                             // Если не хватает
                merge_pass_global<<<pairs, 256>>>(ping ? d_a : d_b, ping ? d_b : d_a, n, run);     // Global merge
            }

            CUDA_CHECK(cudaGetLastError());                       // Проверка ошибки
            CUDA_CHECK(cudaDeviceSynchronize());                  // Синхронизация
            ms_merge_total += t.toc();                            // Суммируем время

            ping = !ping;                                        // Меняем буферы
            run *= 2;                                            // Увеличиваем run
        }

        float ms_total = ms_stage1 + ms_merge_total;             // Общее время

        std::vector<float> out(n);                               // Вектор результата
        CUDA_CHECK(cudaMemcpy(out.data(), (ping ? d_a : d_b), n * sizeof(float), cudaMemcpyDeviceToHost)); // Копирование

        std::cout << std::fixed << std::setprecision(4);         // Формат вывода
        std::cout << "Время этапа 1 ...: " << ms_stage1 << " мс\n"; // Печать времени 1
        std::cout << "Время merge-проходов ...: " << ms_merge_total << " мс\n"; // Печать merge
        std::cout << "Общее время сортировки: " << ms_total << " мс\n"; // Печать итога

        std::cout << "Проверка (CPU): массив отсортирован? "
                  << (is_sorted_cpu(out) ? "ДА" : "НЕТ") << "\n"; // Проверка результата

        std::cout << "Пояснение:\n";                              // Заголовок пояснения
        std::cout << "  • Глобальная память хранит весь массив.\n"; // Пояснение 1
        std::cout << "  • Каждый поток берёт K элементов ...\n";     // Пояснение 2
        std::cout << "  • Слияние выполняется в shared memory...\n"; // Пояснение 3
        std::cout << "  • При больших run ... используется merge без shared.\n\n"; // Пояснение 4

        cudaFree(d_a);                                           // Освобождение d_a
        cudaFree(d_b);                                           // Освобождение d_b
    }

    std::cout << "Готово.\n";                                     // Завершение
    return 0;                                                     // Код успешного выхода
}
