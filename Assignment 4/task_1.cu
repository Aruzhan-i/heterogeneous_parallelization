#include <stdio.h>                     // Подключаем стандартную библиотеку ввода/вывода (printf и т.д.)
#include <cuda_runtime.h>              // Подключаем CUDA Runtime API (cudaMalloc, cudaMemcpy, cudaEvent...)
#include <time.h>                      // Подключаем time.h для srand(time(NULL)) и clock()

#define ARRAY_SIZE 100000              // Макрос: размер массива (количество элементов)
#define THREADS_PER_BLOCK 256          // Макрос: число потоков в одном CUDA-блоке

// CUDA kernel для редукции (суммирования)   // Комментарий: ядро CUDA выполняет редукцию суммы элементов
__global__ void sum_reduction(float *input, float *output, int n) { // Объявление CUDA kernel: input=массив, output=частичные суммы, n=размер
    extern __shared__ float shared_data[];                          // Динамический shared memory (размер задаётся при запуске kernel)
    
    int tid = threadIdx.x;                                          // Индекс потока внутри блока
    int i = blockIdx.x * blockDim.x + threadIdx.x;                  // Глобальный индекс элемента массива для текущего потока
    
    // Загрузка данных в shared memory                              // Комментарий: сначала каждый поток кладёт свой элемент в shared memory
    if (i < n) {                                                    // Проверяем, что индекс не выходит за пределы массива
        shared_data[tid] = input[i];                                // Загружаем input[i] в shared memory по индексу потока
    } else {                                                        // Если поток соответствует индексу за пределами n
        shared_data[tid] = 0.0f;                                    // Записываем 0, чтобы не влиять на сумму
    }
    __syncthreads();                                                // Синхронизация потоков: все должны завершить загрузку shared_data
    
    // Редукция в пределах блока                                     // Комментарий: бинарная редукция (половинный шаг)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {   // stride уменьшается в 2 раза: blockDim/2, /4, /8 ...
        if (tid < stride && i + stride < n) {                       // Только потоки первой половины выполняют сложение, плюс дополнительная проверка границы
            shared_data[tid] += shared_data[tid + stride];          // Добавляем элемент из второй половины shared memory
        }
        __syncthreads();                                            // Синхронизация после каждого шага редукции
    }
    
    // Первый поток записывает результат блока                       // Комментарий: итог суммы блока находится в shared_data[0]
    if (tid == 0) {                                                 // Только поток 0 в каждом блоке
        output[blockIdx.x] = shared_data[0];                        // Записывает сумму блока в выходной массив
    }
}                                                                   // Конец kernel

// CPU версия для сравнения                                         // Комментарий: последовательное суммирование на CPU
float sum_cpu(float *data, int n) {                                 // Функция CPU-суммирования: data=массив, n=размер
    float sum = 0.0f;                                               // Инициализация суммы
    for (int i = 0; i < n; i++) {                                   // Цикл по всем элементам массива
        sum += data[i];                                             // Добавляем текущий элемент к сумме
    }
    return sum;                                                     // Возвращаем итоговую сумму
}                                                                   // Конец CPU функции

int main() {                                                        // Точка входа программы
    int n = ARRAY_SIZE;                                             // Локальная переменная n = размер массива
    size_t size = n * sizeof(float);                                // Размер памяти в байтах для n элементов float
    
    // Выделение памяти на host                                      // Комментарий: память в RAM
    float *h_input = (float*)malloc(size);                          // Выделяем память на host под входной массив
    
    // Инициализация массива случайными значениями                   // Комментарий: заполняем массив данными
    srand(time(NULL));                                              // Инициализируем генератор случайных чисел текущим временем
    for (int i = 0; i < n; i++) {                                   // Цикл по всем элементам
        h_input[i] = (float)(rand() % 100) / 10.0f;                 // Генерируем значение: 0..9.9 (в float)
    }
    
    printf("=== Вычисление суммы %d элементов ===\n\n", n);          // Выводим заголовок с количеством элементов
    
    // ============ CPU вычисление ============                      // Раздел: CPU расчёт
    clock_t cpu_start = clock();                                    // Фиксируем время начала CPU вычислений
    float cpu_sum = sum_cpu(h_input, n);                            // Вычисляем сумму массива на CPU
    clock_t cpu_end = clock();                                      // Фиксируем время окончания CPU вычислений
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // Переводим ticks в миллисекунды
    
    printf("CPU результат:\n");                                     // Вывод заголовка CPU результата
    printf("  Сумма: %.2f\n", cpu_sum);                             // Печатаем сумму CPU с точностью 2 знака
    printf("  Время: %.4f мс\n\n", cpu_time);                       // Печатаем время CPU в мс
    
    // ============ GPU вычисление ============                      // Раздел: GPU расчёт
    float *d_input, *d_output;                                      // Указатели на device память (GPU): вход и выход
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // Кол-во блоков: округление вверх
    size_t output_size = num_blocks * sizeof(float);                // Размер массива выходных данных: по 1 float на блок
    float *h_output = (float*)malloc(output_size);                  // Выделяем host память под результаты блоков
    
    // Выделение памяти на device                                    // Комментарий: память на GPU
    cudaMalloc(&d_input, size);                                     // Выделяем память под входной массив на GPU
    cudaMalloc(&d_output, output_size);                             // Выделяем память под выходной массив (частичные суммы)
    
    // Копирование данных на GPU                                     // Комментарий: передаём входной массив на устройство
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);     // Копируем host->device
    
    // Создание событий для измерения времени                        // Комментарий: CUDA Events для тайминга kernel
    cudaEvent_t start, stop;                                        // Объявляем два CUDA события
    cudaEventCreate(&start);                                        // Создаём событие start
    cudaEventCreate(&stop);                                         // Создаём событие stop
    
    // Запуск kernel с измерением времени                            // Комментарий: записываем start/stop вокруг kernel
    cudaEventRecord(start);                                         // Записываем событие начала (в очередь GPU)
    sum_reduction<<<num_blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_output, n); // Запуск kernel: blocks, threads, shared mem bytes
    cudaEventRecord(stop);                                          // Записываем событие окончания (в очередь GPU)
    
    // Копирование результата обратно                                // Комментарий: получаем частичные суммы блоков
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost); // Копируем device->host
    
    // Финальная редукция на CPU (суммирование результатов блоков)   // Комментарий: складываем суммы блоков на CPU
    float gpu_sum = 0.0f;                                           // Инициализация итоговой GPU суммы
    for (int i = 0; i < num_blocks; i++) {                          // Цикл по всем блокам
        gpu_sum += h_output[i];                                     // Складываем результат каждого блока
    }
    
    // Ожидание завершения и получение времени                       // Комментарий: ждём stop и считаем elapsed time
    cudaEventSynchronize(stop);                                     // Ждём завершения GPU до события stop
    float gpu_time = 0;                                             // Переменная под GPU время (в мс)
    cudaEventElapsedTime(&gpu_time, start, stop);                   // Вычисляем время между start и stop
    
    printf("GPU результат:\n");                                     // Вывод заголовка GPU результата
    printf("  Сумма: %.2f\n", gpu_sum);                             // Печатаем GPU сумму (2 знака)
    printf("  Время: %.4f мс\n", gpu_time);                         // Печатаем GPU время выполнения kernel
    printf("  Количество блоков: %d\n", num_blocks);                // Печатаем число блоков
    printf("  Потоков на блок: %d\n\n", THREADS_PER_BLOCK);         // Печатаем threads/block
    
    // Сравнение результатов                                         // Комментарий: сравниваем CPU и GPU суммы
    float diff = fabs(cpu_sum - gpu_sum);                           // Абсолютная разница между результатами
    printf("=== Анализ ===\n");                                     // Заголовок анализа
    printf("Разница результатов: %.6f\n", diff);                    // Выводим разницу
    printf("Ускорение: %.2fx\n", cpu_time / gpu_time);              // Выводим ускорение CPU/GPU
    
    if (diff < 0.01f) {                                             // Если разница мала (считаем, что совпадает)
        printf("✓ Результаты совпадают!\n");                        // Сообщаем, что результаты совпадают
    } else {                                                        // Иначе
        printf("✗ Результаты различаются (возможны ошибки округления)\n"); // Сообщаем про расхождение/округление
    }
    
    // Освобождение памяти                                           // Комментарий: освобождаем все ресурсы
    free(h_input);                                                  // Освобождаем host входной массив
    free(h_output);                                                 // Освобождаем host выходной массив
    cudaFree(d_input);                                              // Освобождаем device входной массив
    cudaFree(d_output);                                             // Освобождаем device выходной массив
    cudaEventDestroy(start);                                        // Удаляем CUDA событие start
    cudaEventDestroy(stop);                                         // Удаляем CUDA событие stop
    
    return 0;                                                       // Возврат успешного завершения
}                                                                   // Конец main
