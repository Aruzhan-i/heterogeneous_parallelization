// task_1_reduction.cu                                                     // Имя файла: CUDA редукция суммы
// CUDA редукция (сумма массива) с использованием shared memory + проверка на CPU // Описание: редукция + CPU check

#include <cuda_runtime.h>                                                  // CUDA runtime API
#include <iostream>                                                        // std::cout, std::cerr
#include <vector>                                                          // std::vector
#include <numeric>                                                         // (не используется напрямую, но обычно для accumulate)
#include <iomanip>                                                         // std::setprecision, std::fixed
#include <cstdlib>                                                         // std::exit

#define CUDA_CHECK(call)                                                   /* Макрос для проверки ошибок CUDA */ \
    do {                                                                   /* Начало безопасного блока */ \
        cudaError_t err = (call);                                          /* Выполнить CUDA вызов и сохранить код ошибки */ \
        if (err != cudaSuccess) {                                          /* Если ошибка */ \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         /* Печать строки ошибки */ \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";   /* Печать файла и строки */ \
            std::exit(1);                                                  /* Завершение программы с кодом 1 */ \
        }                                                                  /* Конец if */ \
    } while (0)                                                            /* Конец конструкции макроса */

// Ядро: каждый блок суммирует свой кусок массива -> записывает 1 число в out[blockIdx.x] // Назначение CUDA-ядра
__global__ void reduce_sum_shared(const float* __restrict__ in,             // Входной массив на GPU
                                  float* __restrict__ out,                 // Выходной массив: частичные суммы по блокам
                                  int n)                                   // Размер входного массива
{                                                                           // Начало ядра
    extern __shared__ float sdata[];                                        // Shared memory (размер задаём при запуске ядра)

    unsigned int tid = threadIdx.x;                                         // ID потока внутри блока
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;         // Глобальный индекс элемента, учитывая 2 элемента на поток

    // 1) Загружаем данные в shared: сразу по 2 элемента на поток (если есть) // Стадия загрузки данных
    float sum = 0.0f;                                                      // Локальная сумма потока
    if (i < (unsigned)n) sum += in[i];                                      // Добавить первый элемент, если индекс валиден
    if (i + blockDim.x < (unsigned)n) sum += in[i + blockDim.x];            // Добавить второй элемент, если он существует

    sdata[tid] = sum;                                                      // Записать сумму потока в shared memory
    __syncthreads();                                                       // Синхронизация: все потоки загрузили данные

    // 2) Редукция внутри блока в shared memory (дерево суммирования)        // Редукция внутри shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {                 // Шаг редукции: делим активные потоки пополам
        if (tid < s) {                                                     // Только потоки в первой половине выполняют суммирование
            sdata[tid] += sdata[tid + s];                                   // Прибавляем элемент из второй половины
        }                                                                  // Конец if
        __syncthreads();                                                   // Синхронизация после каждого шага
    }                                                                      // Конец цикла редукции

    // 3) Поток 0 пишет результат блока                                      // Запись результата блока
    if (tid == 0) {                                                        // Только поток 0
        out[blockIdx.x] = sdata[0];                                         // Записать сумму блока в out
    }                                                                      // Конец if
}                                                                           // Конец ядра

// Хост-функция: многопроходная редукция до одного числа                     // Функция на CPU, которая управляет редукцией на GPU
float gpu_reduce_sum(const std::vector<float>& h_in, int blockSize = 256)   // Вход: host-вектор, blockSize по умолчанию 256
{                                                                           // Начало функции
    const int n = (int)h_in.size();                                         // Размер входного массива

    float* d_in = nullptr;                                                 // Указатель на входной массив на GPU
    float* d_out = nullptr;                                                // Указатель на выходной массив (частичные суммы) на GPU

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));                       // Выделить память на GPU под вход
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // Копировать входные данные CPU->GPU

    int curN = n;                                                          // Текущий размер массива для редукции
    float* curIn = d_in;                                                   // Текущий входной буфер на GPU

    // будем запускать редукцию, пока не останется 1 элемент                 // Пока массив не сведён к одному значению
    while (curN > 1) {                                                     // Цикл многопроходной редукции
        int gridSize = (curN + (blockSize * 2 - 1)) / (blockSize * 2);      // Количество блоков (каждый блок обрабатывает 2*blockSize элементов)
        CUDA_CHECK(cudaMalloc(&d_out, gridSize * sizeof(float)));           // Выделить память под частичные суммы

        size_t shmBytes = blockSize * sizeof(float);                        // Shared memory: по одному float на поток
        reduce_sum_shared<<<gridSize, blockSize, shmBytes>>>(curIn, d_out, curN); // Запуск CUDA ядра редукции
        CUDA_CHECK(cudaGetLastError());                                     // Проверка ошибок запуска ядра
        CUDA_CHECK(cudaDeviceSynchronize());                                // Дожидаемся завершения ядра

        // освобождаем старый вход (кроме самого первого d_in)               // Освобождение промежуточных буферов
        if (curIn != d_in) CUDA_CHECK(cudaFree(curIn));                     // Освободить предыдущий временный вход, если это не d_in

        // следующий проход                                                  // Подготовка к следующей итерации
        curIn = d_out;                                                     // Новый вход = выход предыдущей редукции
        curN = gridSize;                                                   // Новый размер массива = число блоков
        d_out = nullptr;                                                   // Сброс указателя, чтобы корректно выделить память снова
    }                                                                      // Конец while

    float h_sum = 0.0f;                                                    // Переменная на CPU для результата
    CUDA_CHECK(cudaMemcpy(&h_sum, curIn, sizeof(float), cudaMemcpyDeviceToHost)); // Копировать финальную сумму GPU->CPU

    if (curIn != d_in) CUDA_CHECK(cudaFree(curIn));                         // Освободить финальный временный буфер, если он не d_in
    CUDA_CHECK(cudaFree(d_in));                                             // Освободить исходный буфер входа на GPU

    return h_sum;                                                           // Вернуть сумму
}                                                                           // Конец функции gpu_reduce_sum

int main()                                                                  // Точка входа в программу
{                                                                           // Начало main
    // Тестовый массив (можешь менять размер)                                // Комментарий: настройки теста
    const int N = 100000; // попробуй 16, 1024, 1000000 и т.д.              // Размер массива
    std::vector<float> h(N);                                                // Вектор на CPU размера N

    // Заполняем предсказуемыми значениями                                   // Инициализация массива
    for (int i = 0; i < N; ++i) h[i] = 1.0f; // сумма должна быть N         // Заполнение единицами

    // CPU-эталон                                                            // CPU вычисление суммы для проверки
    double cpu_sum = 0.0;                                                   // CPU сумма (double для точности)
    for (float x : h) cpu_sum += (double)x;                                 // Суммируем все элементы

    // GPU                                                                   // Вызов GPU редукции
    float gpu_sum = gpu_reduce_sum(h, 256);                                  // Вычисление суммы на GPU

    // Проверка                                                              // Проверяем отличие
    double diff = std::abs(cpu_sum - (double)gpu_sum);                      // Абсолютная разница CPU и GPU

    std::cout << std::fixed << std::setprecision(6);                        // Формат вывода: фиксированная точность 6 знаков
    std::cout << "N = " << N << "\n";                                       // Печать размера массива
    std::cout << "CPU sum = " << cpu_sum << "\n";                           // Печать CPU суммы
    std::cout << "GPU sum = " << gpu_sum << "\n";                           // Печать GPU суммы
    std::cout << "Abs diff = " << diff << "\n";                             // Печать абсолютной разницы

    // Допуск (из-за float)                                                  // Т.к. GPU float может дать небольшую погрешность
    const double eps = 1e-2;                                                // Допустимая погрешность
    if (diff < eps) std::cout << "✅ OK: результат корректный\n";           // Если diff маленький — всё хорошо
    else           std::cout << "❌ FAIL: результат отличается\n";           // Иначе считаем ошибкой

    return 0;                                                               // Успешное завершение программы
}                                                                           // Конец main
