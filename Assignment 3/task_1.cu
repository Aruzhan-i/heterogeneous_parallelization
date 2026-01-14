#include <cuda_runtime.h>                             // Подключение CUDA Runtime API
#include <iostream>                                  // Стандартный ввод/вывод C++
#include <vector>                                    // Контейнер std::vector
#include <iomanip>                                   // Форматированный вывод
#include <cmath>                                     // Математические функции (fabs и др.)

#ifdef _WIN32                                        // Проверка: если код компилируется под Windows
  #include <windows.h>                               // Подключение Windows API
#endif                                               // Конец условия для Windows

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) do {                          /* Начало макроса-обёртки */ \
    cudaError_t err = (call);                          /* Выполнение CUDA-вызова и сохранение кода ошибки */ \
    if (err != cudaSuccess) {                          /* Проверка: произошла ли ошибка */ \
        std::cerr << "CUDA error: "                    /* Начало вывода сообщения об ошибке */ \
                  << cudaGetErrorString(err)           /* Текстовое описание ошибки CUDA */ \
                  << " at " << __FILE__                /* Имя файла, где возникла ошибка */ \
                  << ":" << __LINE__                   /* Номер строки, где возникла ошибка */ \
                  << std::endl;                        /* Перевод строки */ \
        std::exit(1);                                  /* Завершение программы с кодом ошибки */ \
    }                                                  /* Конец блока if */ \
} while(0)                                             /* Макрос оформлен как безопасный блок */

// =======================
// 1) Только глобальная память                          // Заголовок: версия с глобальной памятью
// =======================
__global__ void mul_global(const float* __restrict__ in, // CUDA-ядро: указатель на входной массив
                           float* __restrict__ out,     // CUDA-ядро: указатель на выходной массив
                           float k, int n)              // Множитель k и размер массива n
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // Глобальный индекс элемента
    if (idx < n) out[idx] = in[idx] * k;                // Если индекс в пределах — умножаем элемент
}

// =======================
// 2) Shared memory                                      // Заголовок: версия с shared-памятью
// =======================
__global__ void mul_shared(const float* __restrict__ in,// CUDA-ядро: входной массив
                           float* __restrict__ out,    // CUDA-ядро: выходной массив
                           float k, int n)             // Множитель и размер
{
    extern __shared__ float sh[];                       // Объявление динамической shared-памяти
    int g = blockIdx.x * blockDim.x + threadIdx.x;      // Глобальный индекс текущего потока

    if (g < n) sh[threadIdx.x] = in[g];                 // Копируем данные из global в shared
    __syncthreads();                                   // Синхронизация потоков в блоке

    if (g < n) sh[threadIdx.x] *= k;                    // Умножаем элемент в shared-памяти
    __syncthreads();                                   // Снова синхронизация потоков

    if (g < n) out[g] = sh[threadIdx.x];                // Записываем результат обратно в global
}

// Универсальный таймер для kernel                       // Шаблонная функция замера времени ядра
template <typename Kernel, typename... Args>            // Шаблон: тип ядра и список аргументов
float time_kernel_avg(Kernel kernel,                    // Передаваемое CUDA-ядро
                      dim3 grid, dim3 block,            // Размеры сетки и блока
                      size_t shmem_bytes,               // Размер shared-памяти в байтах
                      int warmup, int iters,            // Количество прогревов и итераций замера
                      Args... args)                     // Аргументы ядра
{
    // Прогрев (важно для честного замера)               // Комментарий: зачем нужен warm-up
    for (int i = 0; i < warmup; ++i) {                  // Цикл прогрева
        kernel<<<grid, block, shmem_bytes>>>(args...);  // Запуск ядра без замера времени
    }
    CUDA_CHECK(cudaGetLastError());                     // Проверка последней ошибки CUDA
    CUDA_CHECK(cudaDeviceSynchronize());                // Ожидание завершения всех потоков

    cudaEvent_t start, stop;                            // CUDA-события для измерения времени
    CUDA_CHECK(cudaEventCreate(&start));                // Создание события начала
    CUDA_CHECK(cudaEventCreate(&stop));                 // Создание события конца

    CUDA_CHECK(cudaEventRecord(start));                 // Запись события начала
    for (int i = 0; i < iters; ++i) {                   // Цикл измерений
        kernel<<<grid, block, shmem_bytes>>>(args...);  // Запуск ядра для замера
    }
    CUDA_CHECK(cudaGetLastError());                     // Проверка на ошибки после запусков
    CUDA_CHECK(cudaEventRecord(stop));                  // Запись события конца
    CUDA_CHECK(cudaEventSynchronize(stop));             // Ожидание завершения события

    float ms = 0.0f;                                    // Переменная для времени в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Вычисление прошедшего времени

    CUDA_CHECK(cudaEventDestroy(start));                // Удаление события начала
    CUDA_CHECK(cudaEventDestroy(stop));                 // Удаление события конца

    return ms / iters;                                  // Среднее время одного запуска (мс)
}

int main()                                              // Точка входа в программу
{
#ifdef _WIN32                                           // Если программа под Windows
    // Делает консоль Windows UTF-8 (чтобы русский печатался нормально)
    SetConsoleOutputCP(CP_UTF8);                         // Установка кодировки UTF-8 для консоли
#endif                                                   // Конец условия для Windows

    const int N = 1'000'000;                             // Размер массива (1 миллион элементов)
    const float k = 2.5f;                               // Множитель для умножения

    std::vector<float> h_in(N),                          // Входной массив на хосте
                       h_out_g(N),                      // Выход для global-версии
                       h_out_s(N);                      // Выход для shared-версии
    for (int i = 0; i < N; ++i) h_in[i] = i * 0.001f;    // Инициализация входных данных

    float *d_in = nullptr, *d_out = nullptr;            // Указатели на память устройства
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));  // Выделение памяти под вход на GPU
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));  // Выделение памяти под выход на GPU

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),             // Копирование входных данных
                          N * sizeof(float),            // Размер копируемых данных
                          cudaMemcpyHostToDevice));    // Направление: Host → Device

    const int BLOCK = 256;                              // Количество потоков в одном блоке
    dim3 block(BLOCK);                                  // Конфигурация блока
    dim3 grid((N + BLOCK - 1) / BLOCK);                 // Количество блоков в сетке

    const int WARMUP = 50;                              // Количество прогревочных запусков
    const int ITERS  = 1000;                            // Количество итераций замера

    // 1) global                                        // Замер версии с глобальной памятью
    float t_global = time_kernel_avg(                   // Вызов функции замера
        mul_global, grid, block,                        // Ядро и конфигурация
        0, WARMUP, ITERS,                               // Shared-память = 0
        d_in, d_out, k, N                               // Аргументы ядра
    );
    CUDA_CHECK(cudaMemcpy(h_out_g.data(), d_out,         // Копирование результата
                          N * sizeof(float),            // Размер данных
                          cudaMemcpyDeviceToHost));    // Направление: Device → Host

    // 2) shared                                        // Замер версии с shared-памятью
    size_t shmem = BLOCK * sizeof(float);               // Размер shared-памяти на блок
    float t_shared = time_kernel_avg(                   // Вызов функции замера
        mul_shared, grid, block,                        // Ядро и конфигурация
        shmem, WARMUP, ITERS,                           // Передаём размер shared-памяти
        d_in, d_out, k, N                               // Аргументы ядра
    );
    CUDA_CHECK(cudaMemcpy(h_out_s.data(), d_out,         // Копирование результата
                          N * sizeof(float),            // Размер данных
                          cudaMemcpyDeviceToHost));    // Направление: Device → Host

    // Проверка корректности (несколько элементов)       // Валидация результатов
    bool ok = true;                                     // Флаг корректности
    for (int i = 0; i < 20; ++i) {                      // Проверяем первые 20 элементов
        float ref = h_in[i] * k;                        // Эталонное значение
        if (std::fabs(h_out_g[i] - ref) > 1e-5f) ok = false; // Проверка global-версии
        if (std::fabs(h_out_s[i] - ref) > 1e-5f) ok = false; // Проверка shared-версии
    }

    std::cout << std::fixed << std::setprecision(6);    // Формат вывода: фиксированный, 6 знаков
    std::cout << "Размер массива: " << N << "\n";      // Вывод размера массива
    std::cout << "Потоков в блоке: " << BLOCK           // Вывод конфигурации блока
              << ", блоков: " << grid.x << "\n";       // Вывод количества блоков
    std::cout << "Замер: warmup=" << WARMUP             // Вывод параметров замера
              << ", iters=" << ITERS << "\n\n";

    std::cout << "Время (глобальная память), среднее: "
              << t_global << " мс\n";                  // Вывод времени global-версии
    std::cout << "Время (shared память),     среднее: "
              << t_shared << " мс\n";                  // Вывод времени shared-версии
    std::cout << "Отношение (global/shared): "
              << (t_global / t_shared) << "x\n";       // Вывод отношения скоростей
    std::cout << "Проверка: " 
              << (ok ? "OK" : "FAILED") << "\n";       // Вывод результата проверки

    CUDA_CHECK(cudaFree(d_in));                         // Освобождение памяти входа на GPU
    CUDA_CHECK(cudaFree(d_out));                        // Освобождение памяти выхода на GPU
    return 0;                                          // Завершение программы
}
