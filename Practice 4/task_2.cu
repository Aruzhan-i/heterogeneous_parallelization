// task_2.cu                                   // Имя файла с CUDA-кодом
// Задание 2: редукция суммы                  // Описание задания: суммирование элементов массива
// (a) Только глобальная память: atomicAdd    // Вариант A: атомарное сложение в глобальной памяти
// (b) Глобальная + разделяемая: редукция     // Вариант B: редукция в shared памяти + один atomicAdd

#include <cuda_runtime.h>                     // Подключаем CUDA Runtime API
#include <iostream>                           // Подключаем ввод/вывод (cout, cerr)
#include <vector>                             // Подключаем контейнер std::vector
#include <random>                             // Подключаем генераторы случайных чисел
#include <iomanip>                            // Подключаем форматирование вывода
#include <cstdlib>                            // Подключаем стандартные функции (exit и др.)

// Макрос проверки ошибок CUDA (ВАЖНО: после символа '\' нельзя писать комментарии! поэтому внизу построчный комментарий отдельно)
   /* Выполняем CUDA-вызов и сохраняем код ошибки*/ 
   /* Проверяем, произошла ли ошибка*/ 
     // Выводим строку с описанием ошибки
   // Выводим числовой код ошибки
 // Показываем файл и строку
   // Завершаем программу с ошибкой
   // Конец проверки
// Оборачиваем в do-while для корректного макроса
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "Ошибка CUDA: " << cudaGetErrorString(err)    \
                  << " (код " << (int)err << ") в "               \
                  << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                              \
    }                                                              \
} while(0)

struct GpuTimer {                          // Структура для измерения времени на GPU
    cudaEvent_t start{}, stop{};           // CUDA-события для начала и конца замера
    GpuTimer(){                            // Конструктор таймера
        CUDA_CHECK(cudaEventCreate(&start)); // Создаём событие начала
        CUDA_CHECK(cudaEventCreate(&stop));  // Создаём событие конца
    }
    ~GpuTimer(){                           // Деструктор таймера
        cudaEventDestroy(start);           // Удаляем событие начала
        cudaEventDestroy(stop);            // Удаляем событие конца
    }
    void tic(cudaStream_t s=0){            // Функция начала замера времени
        CUDA_CHECK(cudaEventRecord(start, s)); // Записываем событие старта в поток
    }
    float toc(cudaStream_t s=0){           // Функция окончания замера времени
        CUDA_CHECK(cudaEventRecord(stop, s));     // Записываем событие конца
        CUDA_CHECK(cudaEventSynchronize(stop));   // Ждём завершения события
        float ms=0.f;                             // Переменная для времени в миллисекундах
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Считаем прошедшее время
        return ms;                                 // Возвращаем время
    }
};

// (a) Только глобальная память: атомик на каждый элемент
__global__ void reduce_global_atomic(const float* a, int n, float* out){ // Ядро: редукция через global memory
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if(i < n) atomicAdd(out, a[i]);                // Если индекс в пределах — атомарно прибавляем элемент
}

// (b) Global + shared: редукция в shared и 1 atomicAdd на блок
__global__ void reduce_shared_block(const float* a, int n, float* out){ // Ядро: редукция через shared memory
    extern __shared__ float s[];         // Объявляем динамическую shared память
    int tid = threadIdx.x;              // Локальный индекс потока в блоке
    int i = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента

    s[tid] = (i < n) ? a[i] : 0.f;       // Загружаем данные в shared память или 0
    __syncthreads();                    // Синхронизируем все потоки блока

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){ // Итеративная редукция
        if(tid < stride) s[tid] += s[tid + stride];          // Складываем пары элементов
        __syncthreads();                                     // Синхронизация после шага
    }

    if(tid == 0) atomicAdd(out, s[0]);   // Первый поток блока добавляет сумму в global память
}

static std::vector<float> generate_data(int n){ // Функция генерации входных данных
    std::mt19937 rng(42);                         // Генератор случайных чисел с фиксированным seed
    std::uniform_real_distribution<float> dist(0.f, 1.f); // Распределение [0,1]
    std::vector<float> h(n);                     // Создаём вектор размера n
    for(int i=0;i<n;i++) h[i] = dist(rng);        // Заполняем вектор случайными числами
    return h;                                    // Возвращаем вектор
}

// один прогон -> время в мс, плюс возвращаем итоговую сумму GPU
static float run_once(const std::vector<float>& h, bool use_shared, int block, float& sum_gpu){ // Один запуск теста
    int n = (int)h.size();              // Получаем размер массива
    float *d_a=nullptr, *d_out=nullptr; // Указатели на данные в памяти GPU
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));       // Выделяем память под входной массив
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));       // Выделяем память под результат
    CUDA_CHECK(cudaMemcpy(d_a, h.data(), n*sizeof(float), cudaMemcpyHostToDevice)); // Копируем данные на GPU
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));     // Обнуляем результат на GPU

    int grid = (n + block - 1) / block;  // Вычисляем количество блоков

    GpuTimer t;                          // Создаём таймер
    t.tic();                             // Запускаем таймер
    if(!use_shared){                    // Если используем только global память
        reduce_global_atomic<<<grid, block>>>(d_a, n, d_out); // Запускаем ядро global
    }else{                              // Если используем shared память
        size_t sh = (size_t)block * sizeof(float); // Размер shared памяти
        reduce_shared_block<<<grid, block, sh>>>(d_a, n, d_out); // Запускаем ядро shared
    }
    CUDA_CHECK(cudaGetLastError());      // Проверяем ошибки запуска ядра
    CUDA_CHECK(cudaDeviceSynchronize()); // Ждём завершения вычислений
    float ms = t.toc();                  // Останавливаем таймер и получаем время

    CUDA_CHECK(cudaMemcpy(&sum_gpu, d_out, sizeof(float), cudaMemcpyDeviceToHost)); // Копируем результат на CPU
    cudaFree(d_a);                       // Освобождаем память входного массива на GPU
    cudaFree(d_out);                     // Освобождаем память результата на GPU
    return ms;                           // Возвращаем время выполнения
}

int main(){  // <-- ВАЖНО: точка входа, без неё LNK1561
    std::cout << "=============================================\n"; // Разделитель
    std::cout << "Задание 2: Редукция суммы (global vs shared)\n"; // Заголовок программы
    std::cout << "=============================================\n\n"; // Разделитель

    std::vector<int> sizes  = { 10000, 100000, 1000000 }; // Набор размеров массивов
    std::vector<int> blocks = { 64, 128, 256, 512, 1024 }; // Набор размеров блоков
    const int iters = 20;                                // Количество повторов для усреднения

    std::cout << std::left                              // Выравнивание по левому краю
              << std::setw(12) << "Размер n"            // Заголовок колонки размера
              << std::setw(10) << "Блок"                // Заголовок колонки блока
              << std::setw(18) << "Global (мс)"         // Заголовок времени global
              << std::setw(18) << "Shared (мс)"         // Заголовок времени shared
              << std::setw(12) << "Ускорение"           // Заголовок ускорения
              << "\n";                                  // Перевод строки
    std::cout << std::string(70, '-') << "\n";          // Линия-разделитель

    for(int n : sizes){                                 // Перебираем размеры массивов
        auto h = generate_data(n);                      // Генерируем данные

        float best_g = 1e30f, best_s = 1e30f;           // Лучшие времена для global и shared

        for(int block : blocks){                        // Перебираем размеры блоков
            float g_ms=0.f, s_ms=0.f;                   // Суммарное время для усреднения
            float sum_g=0.f, sum_s=0.f;                 // Результаты сумм

            for(int i=0;i<iters;i++){                   // Запуски для global
                float tmp=0.f;                          // Временная переменная суммы
                g_ms += run_once(h, false, block, tmp); // Запускаем вариант global
                sum_g = tmp;                            // Сохраняем сумму
            }
            g_ms /= iters;                              // Усредняем время

            for(int i=0;i<iters;i++){                   // Запуски для shared
                float tmp=0.f;                          // Временная переменная суммы
                s_ms += run_once(h, true, block, tmp);  // Запускаем вариант shared
                sum_s = tmp;                            // Сохраняем сумму
            }
            s_ms /= iters;                              // Усредняем время

            if(g_ms < best_g) best_g = g_ms;            // Обновляем лучшее время global
            if(s_ms < best_s) best_s = s_ms;            // Обновляем лучшее время shared

            float speedup = g_ms / s_ms;                // Считаем ускорение

            std::cout << std::left                      // Вывод строки таблицы
                      << std::setw(12) << n             // Вывод размера массива
                      << std::setw(10) << block         // Вывод размера блока
                      << std::setw(18) << std::fixed << std::setprecision(4) << g_ms // Время global
                      << std::setw(18) << std::fixed << std::setprecision(4) << s_ms // Время shared
                      << std::setw(12) << std::fixed << std::setprecision(2) << speedup // Ускорение
                      << "\n";                           // Перевод строки
        }

        std::cout << "\nИтог для n=" << n << ":\n";       // Заголовок итогов
        std::cout << "  Лучшее время (только global)  = " << best_g << " мс\n"; // Лучшее время global
        std::cout << "  Лучшее время (global+shared)  = " << best_s << " мс\n"; // Лучшее время shared
        std::cout << "  Ускорение (лучшее) S = " << (best_g / best_s) << "\n\n"; // Лучшее ускорение

        std::cout << "Пояснение: shared-редукция быстрее, потому что суммирование\n" // Поясняющий текст
                  << "внутри блока выполняется в быстрой разделяемой памяти, а atomicAdd\n" // Причина ускорения
                  << "делается один раз на блок (а не для каждого элемента).\n\n"; // Итоговое объяснение
    }

    std::cout << "Готово.\n";        // Сообщение о завершении программы
    return 0;                        // Возвращаем код успешного завершения
}
