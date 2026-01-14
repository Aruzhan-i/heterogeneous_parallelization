#include <cuda_runtime.h>                              // Подключение CUDA Runtime API
#include <iostream>                                   // Стандартный поток ввода/вывода
#include <vector>                                     // Контейнер std::vector
#include <cmath>                                      // Математические функции (fabs и др.)
#include <iomanip>                                    // Форматированный вывод
#include <limits>                                     // Числовые пределы типов

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
} while (0)                                            /* Макрос оформлен как безопасный блок */

// CUDA-ядро для поэлементного сложения векторов
__global__ void vec_add(const float* __restrict__ a,   // Указатель на первый входной массив
                        const float* __restrict__ b,   // Указатель на второй входной массив
                        float* __restrict__ c,         // Указатель на выходной массив
                        int n)                         // Размер массивов
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // Вычисление глобального индекса потока
    if (i < n) c[i] = a[i] + b[i];                     // Если индекс в пределах — складываем элементы
}

// Функция заполнения вектора тестовыми данными
static void fill_data(std::vector<float>& v, float seed) { // Принимает вектор и начальное значение
    for (size_t i = 0; i < v.size(); ++i)              // Проход по всем элементам вектора
        v[i] = seed + 0.001f * static_cast<float>(i % 1000); // Заполнение значениями с небольшим разбросом
}

// Функция проверки корректности результата
static bool check(const std::vector<float>& a,         // Первый входной массив
                  const std::vector<float>& b,         // Второй входной массив
                  const std::vector<float>& c)         // Результирующий массив
{
    for (size_t i = 0; i < c.size(); ++i) {             // Проход по всем элементам результата
        float ref = a[i] + b[i];                        // Эталонное значение
        if (std::fabs(c[i] - ref) > 1e-5f) return false;// Проверка допустимой погрешности
    }
    return true;                                       // Если всё корректно — вернуть true
}

// Функция вывода информации о GPU
static void print_device_info() {
    int dev = 0;                                       // Номер устройства (по умолчанию 0)
    CUDA_CHECK(cudaGetDevice(&dev));                   // Получение текущего устройства
    cudaDeviceProp p{};                                // Структура для свойств GPU
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));      // Получение характеристик GPU

    std::cout << "=== Информация о GPU ===\n";         // Заголовок блока информации
    std::cout << "Устройство: " << dev << "\n";        // Вывод номера устройства
    std::cout << "Название:   " << p.name << "\n";     // Вывод имени видеокарты
    std::cout << "SM:         " << p.multiProcessorCount << "\n"; // Количество SM
    std::cout << "Warp size:  " << p.warpSize << "\n"; // Размер warp
    std::cout << "Max threads/block: " << p.maxThreadsPerBlock << "\n"; // Макс. потоков в блоке
    std::cout << "=== Конец информации о GPU ===\n\n"; // Конец блока информации
}

int main() {                                           // Точка входа в программу
    // Размер массива: 2^24 = 16,777,216 элементов (float)
    const int n = 1 << 24;                             // Количество элементов в массивах
    const size_t bytes = static_cast<size_t>(n) * sizeof(float); // Размер массива в байтах

    // Минимум 3 размера блока (можно оставить 3, но я даю 4 — не запрещено)
    const std::vector<int> block_sizes = {64, 128, 256, 512}; // Возможные размеры блока

    // Параметры замеров
    const int warmup_iters = 5;                        // Количество прогревочных запусков
    const int timed_iters  = 50;                       // Количество итераций для замера

    print_device_info();                               // Вывод информации о GPU

    std::cout << "=== Параметры эксперимента ===\n";   // Заголовок параметров эксперимента
    std::cout << "N (элементов):         " << n << "\n"; // Вывод количества элементов
    std::cout << "Размер одного массива: " 
              << (bytes / (1024.0 * 1024.0)) << " MiB\n"; // Вывод размера массива в MiB
    std::cout << "Прогрев (итераций):    " << warmup_iters << "\n"; // Прогрев
    std::cout << "Замер (итераций):      " << timed_iters << "\n";  // Замер
    std::cout << "Размеры блока:         ";           // Заголовок списка блоков
    for (size_t i = 0; i < block_sizes.size(); ++i) { // Перебор размеров блока
        std::cout << block_sizes[i]                   // Вывод текущего размера блока
                  << (i + 1 < block_sizes.size() ? ", " : "\n"); // Форматирование вывода
    }
    std::cout << "=== Конец параметров ===\n\n";      // Конец блока параметров

    // Хост-данные
    std::vector<float> h_a(n), h_b(n), h_c(n);         // Векторы на стороне CPU
    fill_data(h_a, 1.0f);                              // Заполнение первого массива
    fill_data(h_b, 2.0f);                              // Заполнение второго массива

    // Девайс-буферы
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr; // Указатели на память GPU
    CUDA_CHECK(cudaMalloc(&d_a, bytes));              // Выделение памяти под a на GPU
    CUDA_CHECK(cudaMalloc(&d_b, bytes));              // Выделение памяти под b на GPU
    CUDA_CHECK(cudaMalloc(&d_c, bytes));              // Выделение памяти под c на GPU

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice)); // Копирование a → GPU
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice)); // Копирование b → GPU

    // Таймер
    cudaEvent_t start, stop;                          // CUDA-события для замера времени
    CUDA_CHECK(cudaEventCreate(&start));              // Создание события начала
    CUDA_CHECK(cudaEventCreate(&stop));               // Создание события конца

    std::cout << "=== Результаты замеров (время ядра CUDA) ===\n"; // Заголовок таблицы
    std::cout << std::left
              << std::setw(10) << "Block"             // Колонка: размер блока
              << std::setw(10) << "Grid"              // Колонка: размер сетки
              << std::setw(14) << "Avg ms"            // Колонка: среднее время
              << std::setw(14) << "Min ms"            // Колонка: минимальное время
              << std::setw(14) << "Max ms"            // Колонка: максимальное время
              << std::setw(16) << "Оценка GB/s"       // Колонка: пропускная способность
              << "\n";
    std::cout << std::string(78, '-') << "\n";        // Разделительная линия

    // Для min/max делаем timed_iters отдельных измерений
    for (int bs : block_sizes) {                      // Перебор размеров блока
        int grid = (n + bs - 1) / bs;                 // Расчёт количества блоков

        // Прогрев
        for (int i = 0; i < warmup_iters; ++i) {      // Цикл прогрева
            vec_add<<<grid, bs>>>(d_a, d_b, d_c, n);  // Запуск ядра без замера
        }
        CUDA_CHECK(cudaDeviceSynchronize());          // Ожидание завершения

        float sum_ms = 0.0f;                          // Суммарное время
        float min_ms = std::numeric_limits<float>::infinity(); // Минимальное время
        float max_ms = 0.0f;                          // Максимальное время

        for (int it = 0; it < timed_iters; ++it) {    // Цикл измерений
            CUDA_CHECK(cudaEventRecord(start));       // Запись события начала
            vec_add<<<grid, bs>>>(d_a, d_b, d_c, n);  // Запуск ядра
            CUDA_CHECK(cudaEventRecord(stop));        // Запись события конца
            CUDA_CHECK(cudaEventSynchronize(stop));   // Ожидание завершения

            float ms = 0.0f;                          // Время одного запуска
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Получение времени

            sum_ms += ms;                             // Добавление к сумме
            if (ms < min_ms) min_ms = ms;             // Обновление минимума
            if (ms > max_ms) max_ms = ms;             // Обновление максимума
        }

        float avg_ms = sum_ms / timed_iters;          // Среднее время

        // Оценка пропускной способности
        double gb_moved = (3.0 * static_cast<double>(bytes)) / 1e9; // Передано данных (GB)
        double seconds = avg_ms / 1e3;                // Перевод мс в секунды
        double gbps = gb_moved / seconds;             // GB/s

        std::cout << std::left
                  << std::setw(10) << bs              // Вывод размера блока
                  << std::setw(10) << grid            // Вывод размера сетки
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_ms // Среднее
                  << std::setw(14) << std::fixed << std::setprecision(4) << min_ms // Минимум
                  << std::setw(14) << std::fixed << std::setprecision(4) << max_ms // Максимум
                  << std::setw(16) << std::fixed << std::setprecision(2) << gbps   // GB/s
                  << "\n";
    }

    std::cout << "=== Конец таблицы ===\n\n";         // Конец таблицы результатов

    // Копируем результат и проверяем
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost)); // Копирование c → CPU

    bool ok = check(h_a, h_b, h_c);                   // Проверка корректности
    std::cout << "Проверка корректности: " 
              << (ok ? "OK" : "FAIL") << "\n";        // Вывод результата проверки

    // Печать первых элементов
    std::cout << "\nПервые 10 элементов:\n";          // Заголовок вывода
    std::cout << std::left
              << std::setw(6)  << "i"                 // Колонка индекса
              << std::setw(12) << "a[i]"              // Колонка a[i]
              << std::setw(12) << "b[i]"              // Колонка b[i]
              << std::setw(12) << "c[i]=a+b"          // Колонка результата
              << "\n";
    std::cout << std::string(42, '-') << "\n";        // Разделительная линия
    for (int i = 0; i < 10; ++i) {                    // Вывод первых 10 элементов
        std::cout << std::left
                  << std::setw(6)  << i
                  << std::setw(12) << std::fixed << std::setprecision(4) << h_a[i]
                  << std::setw(12) << std::fixed << std::setprecision(4) << h_b[i]
                  << std::setw(12) << std::fixed << std::setprecision(4) << h_c[i]
                  << "\n";
    }

    // Очистка ресурсов
    CUDA_CHECK(cudaEventDestroy(start));              // Удаление события начала
    CUDA_CHECK(cudaEventDestroy(stop));               // Удаление события конца
    CUDA_CHECK(cudaFree(d_a));                        // Освобождение памяти d_a
    CUDA_CHECK(cudaFree(d_b));                        // Освобождение памяти d_b
    CUDA_CHECK(cudaFree(d_c));                        // Освобождение памяти d_c

    return 0;                                         // Завершение программы
}
