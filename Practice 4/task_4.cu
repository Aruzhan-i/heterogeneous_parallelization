// task_4.cu                                                        // Имя файла CUDA-программы
// Задание 4: Замер времени + построение графиков (SVG) без внешних зависимостей // Описание задания
// Выход:                                                           // Указание выходных файлов
//   results/times.csv                                               // CSV файл со временем
//   results/reduce_times.svg                                        // SVG график для редукции
//   results/sort_times.svg                                          // SVG график для сортировки

#include <cuda_runtime.h>                                           // CUDA Runtime API
#include <iostream>                                                 // Потоки ввода/вывода
#include <vector>                                                   // std::vector
#include <random>                                                   // Генерация случайных чисел
#include <iomanip>                                                  // Форматирование вывода
#include <fstream>                                                  // Работа с файлами
#include <cstdlib>                                                  // std::system, std::exit
#include <string>                                                   // std::string
#include <algorithm>                                                // std::max, min, std::string operations
#include <sstream>   // <-- ВАЖНО: для std::ostringstream           // Потоки для сборки строк

// Макрос проверки ошибок CUDA (ВАЖНО: в строках с '\' нельзя дописывать комментарии после '\') // Пояснение правила макроса
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "Ошибка CUDA: " << cudaGetErrorString(err)    \
                  << " (код " << (int)err << ") в "               \
                  << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                              \
    }                                                              \
} while(0)

// Таймер на GPU (cudaEvent)                                         // Комментарий к таймеру
struct GpuTimer {                                                   // Объявление структуры таймера
    cudaEvent_t start{}, stop{};                                    // CUDA события: старт и стоп
    GpuTimer(){ CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); } // Конструктор: создаём события
    ~GpuTimer(){ cudaEventDestroy(start); cudaEventDestroy(stop); } // Деструктор: удаляем события
    void tic(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(start, s)); } // Запуск таймера (запись start)
    float toc(cudaStream_t s=0){                                    // Остановка таймера и возврат времени
        CUDA_CHECK(cudaEventRecord(stop, s));                       // Запись stop события
        CUDA_CHECK(cudaEventSynchronize(stop));                     // Ожидание stop события
        float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Подсчёт времени
        return ms;                                                  // Возвращаем миллисекунды
    }                                                               // Конец toc()
};                                                                  // Конец структуры GpuTimer

// Создание папки results (Windows/Linux), без <filesystem>          // Комментарий: создаём папку results
static void make_results_dir() {                                    // Функция создания папки
#ifdef _WIN32                                                       // Если Windows
    std::system("if not exist results mkdir results >nul 2>nul");   // Создаём results (скрываем вывод)
#else                                                               // Иначе (Linux/macOS)
    std::system("mkdir -p results >/dev/null 2>&1");                // Создаём results (скрываем вывод)
#endif                                                              // Конец ветвления по ОС
}                                                                   // Конец make_results_dir()

// =====================================================             // Разделитель
// Task 2: Редукция суммы                                            // Раздел редукции
// =====================================================             // Разделитель

// (a) Только global: atomicAdd на каждый элемент                     // Вариант A: atomicAdd для каждого элемента
__global__ void reduce_global_atomic(const float* __restrict__ a, int n, float* out){ // CUDA-ядро редукции (global)
    int i = blockIdx.x * blockDim.x + threadIdx.x;                  // Глобальный индекс элемента
    if(i < n) atomicAdd(out, a[i]);                                 // Если в пределах — атомарно добавляем
}                                                                   // Конец reduce_global_atomic

// (b) Global + shared: редукция в shared + 1 atomicAdd на блок       // Вариант B: редукция в shared + 1 atomicAdd
__global__ void reduce_shared_block(const float* __restrict__ a, int n, float* out){ // CUDA-ядро редукции (shared)
    extern __shared__ float s[];                                    // Динамическая shared память
    int tid = threadIdx.x;                                          // Индекс потока в блоке
    int i = blockIdx.x * blockDim.x + tid;                          // Глобальный индекс элемента

    s[tid] = (i < n) ? a[i] : 0.f;                                  // Загружаем элемент или 0
    __syncthreads();                                                // Синхронизация блока

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){       // Параллельная редукция
        if(tid < stride) s[tid] += s[tid + stride];                 // Складываем пары
        __syncthreads();                                            // Синхронизация после шага
    }                                                               // Конец цикла редукции

    if(tid == 0) atomicAdd(out, s[0]);                              // Поток 0 добавляет сумму блока
}                                                                   // Конец reduce_shared_block

// Один прогон редукции -> ms                                        // Комментарий: функция одного прогона редукции
static float run_reduce_once(const std::vector<float>& h, bool use_shared, int block){ // Запуск редукции один раз
    int n = (int)h.size();                                          // Размер входного массива
    float *d_a=nullptr, *d_out=nullptr;                             // Указатели на память GPU
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));                  // Выделяем память под вход
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));                  // Выделяем память под результат

    CUDA_CHECK(cudaMemcpy(d_a, h.data(), n*sizeof(float), cudaMemcpyHostToDevice)); // Копируем данные на GPU
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));                // Обнуляем выход

    int grid = (n + block - 1) / block;                             // Количество блоков в сетке

    GpuTimer t;                                                     // Таймер GPU
    t.tic();                                                        // Старт таймера
    if(!use_shared){                                                // Если выбран вариант global
        reduce_global_atomic<<<grid, block>>>(d_a, n, d_out);        // Запускаем global-ядро
    } else {                                                        // Иначе используем shared
        size_t sh = (size_t)block * sizeof(float);                  // Размер shared памяти
        reduce_shared_block<<<grid, block, sh>>>(d_a, n, d_out);     // Запускаем shared-ядро
    }                                                               // Конец выбора варианта
    CUDA_CHECK(cudaGetLastError());                                 // Проверка ошибок запуска ядра
    CUDA_CHECK(cudaDeviceSynchronize());                            // Ожидание завершения
    float ms = t.toc();                                             // Получаем время выполнения

    cudaFree(d_a);                                                  // Освобождаем d_a
    cudaFree(d_out);                                                // Освобождаем d_out
    return ms;                                                      // Возвращаем время в мс
}                                                                   // Конец run_reduce_once

// =====================================================             // Разделитель
// Task 3: Сортировка (local bubble + shared merge + fallback)       // Раздел сортировки
// =====================================================             // Разделитель
constexpr int   K   = 8;                                            // Сколько элементов на поток
constexpr float INF = 1e30f;                                        // Бесконечность для заполнения хвоста

__device__ __forceinline__ void bubble_sort_local(float v[K]) {     // Локальная пузырьковая сортировка K элементов
    #pragma unroll                                                  // Просим компилятор развернуть цикл
    for (int i = 0; i < K; i++) {                                   // Внешний цикл
        #pragma unroll                                              // Развернуть внутренний цикл
        for (int j = 0; j < K - 1 - i; j++) {                       // Внутренний цикл
            if (v[j] > v[j + 1]) {                                  // Если не по возрастанию
                float t = v[j];                                     // Временная переменная
                v[j] = v[j + 1];                                    // Обмен
                v[j + 1] = t;                                       // Обмен
            }                                                       // Конец if
        }                                                           // Конец внутреннего цикла
    }                                                               // Конец внешнего цикла
}                                                                   // Конец bubble_sort_local

__global__ void sort_tile_local_then_merge_shared(float* a, int n) { // Ядро: сортировка тайла + merge в shared
    extern __shared__ float sh[]; // 2*tile                          // Shared буфер: данные + tmp
    int tid  = threadIdx.x;                                          // Индекс потока
    int tile = blockDim.x * K;                                       // Размер тайла

    float* s   = sh;                                                 // Указатель на основную часть shared
    float* tmp = sh + tile;                                          // Указатель на временный буфер

    int tile_base = blockIdx.x * tile;                               // Смещение тайла в global

    float loc[K];                                                    // Локальный массив потока
    int base = tile_base + tid * K;                                  // База для чтения K элементов

    #pragma unroll                                                   // Разворачиваем цикл чтения
    for(int i=0;i<K;i++){                                            // Читаем K элементов
        int idx = base + i;                                          // Индекс в global
        loc[i] = (idx < n) ? a[idx] : INF;                           // Читаем или INF
    }                                                                // Конец чтения

    bubble_sort_local(loc);                                          // Сортируем локально

    #pragma unroll                                                   // Разворачиваем цикл записи
    for(int i=0;i<K;i++) s[tid*K + i] = loc[i];                      // Пишем отсортированный кусок в shared
    __syncthreads();                                                 // Синхронизация

    for(int run = K; run < tile; run *= 2){                          // Merge в shared с увеличением run
        for(int start = tid * (2*run); start < tile; start += blockDim.x * (2*run)){ // Пакеты длины 2*run
            int mid = start + run;                                   // Середина
            int end = min(start + 2*run, tile);                      // Конец (не дальше tile)

            int i = start, j = mid, t = start;                       // Индексы для merge
            while(i < mid && j < end) tmp[t++] = (s[i] <= s[j]) ? s[i++] : s[j++]; // Основной merge
            while(i < mid) tmp[t++] = s[i++];                        // Хвост первой части
            while(j < end) tmp[t++] = s[j++];                        // Хвост второй части
        }                                                            // Конец обработки пакетов
        __syncthreads();                                             // Синхронизация

        for(int idx = tid; idx < tile; idx += blockDim.x) s[idx] = tmp[idx]; // Копируем tmp -> s
        __syncthreads();                                             // Синхронизация
    }                                                                // Конец цикла run

    for(int idx = tid; idx < tile; idx += blockDim.x){               // Запись тайла обратно в global
        int g = tile_base + idx;                                     // Глобальный индекс
        if(g < n) a[g] = s[idx];                                     // Запись если в пределах
    }                                                                // Конец записи тайла
}                                                                    // Конец sort_tile_local_then_merge_shared

__global__ void merge_pass_shared(const float* in, float* out, int n, int run) { // Merge-проход в shared
    extern __shared__ float s[]; // 2*run                             // Shared буфер для A и B
    int base = blockIdx.x * 2 * run;                                  // База пары

    float* A = s;                                                     // A в shared
    float* B = s + run;                                               // B в shared

    for(int i=threadIdx.x; i<run; i+=blockDim.x){                     // Загрузка A и B в shared
        int ia = base + i;                                            // Индекс A в global
        int ib = base + run + i;                                      // Индекс B в global
        A[i] = (ia < n) ? in[ia] : INF;                               // Читаем A или INF
        B[i] = (ib < n) ? in[ib] : INF;                               // Читаем B или INF
    }                                                                 // Конец загрузки
    __syncthreads();                                                  // Синхронизация

    if(threadIdx.x == 0){                                             // Только поток 0 делает merge (учебно)
        int i=0, j=0;                                                 // Индексы в A и B
        for(int k=0;k<2*run;k++){                                     // Записываем 2*run элементов
            float x = (A[i] <= B[j]) ? A[i++] : B[j++];               // Берём меньший
            int outIdx = base + k;                                    // Индекс в выходе
            if(outIdx < n) out[outIdx] = x;                           // Записываем если в пределах
        }                                                             // Конец k
    }                                                                 // Конец if thread0
}                                                                    // Конец merge_pass_shared

__device__ __forceinline__ float getA(const float* in, int base, int i, int run, int n) { // Безопасный доступ к A
    int idx = base + i;                                               // Индекс в global
    return (i >= 0 && i < run && idx < n) ? in[idx] : INF;            // Возвращаем значение или INF
}                                                                    // Конец getA
__device__ __forceinline__ float getB(const float* in, int base, int i, int run, int n) { // Безопасный доступ к B
    int idx = base + run + i;                                         // Индекс в global
    return (i >= 0 && i < run && idx < n) ? in[idx] : INF;            // Возвращаем значение или INF
}                                                                    // Конец getB

__global__ void merge_pass_global(const float* in, float* out, int n, int run) { // Merge без shared (merge-path)
    int base = blockIdx.x * 2 * run;                                  // База пары

    for(int k = threadIdx.x; k < 2*run; k += blockDim.x){             // Каждой нити — несколько k
        int i_min = max(0, k - run);                                  // Нижняя граница i
        int i_max = min(k, run);                                      // Верхняя граница i

        while(i_min < i_max){                                         // Бинарный поиск i
            int i = (i_min + i_max) / 2;                              // Середина
            int j = k - i;                                            // Соответствующий j

            float A_i   = getA(in, base, i,     run, n);              // A[i]
            float B_jm1 = getB(in, base, j - 1, run, n);              // B[j-1]

            if(A_i < B_jm1) i_min = i + 1;                            // Сдвиг вправо
            else i_max = i;                                           // Сдвиг влево
        }                                                             // Конец бинарного поиска

        int i = i_min;                                                // Итоговый i
        int j = k - i;                                                // Итоговый j

        float A_im1 = getA(in, base, i - 1, run, n);                  // A[i-1]
        float B_jm1 = getB(in, base, j - 1, run, n);                  // B[j-1]

        float x = (A_im1 > B_jm1) ? A_im1 : B_jm1;                    // Выбор элемента merged

        int outIdx = base + k;                                        // Индекс записи
        if(outIdx < n) out[outIdx] = x;                               // Записываем если в пределах
    }                                                                 // Конец цикла k
}                                                                    // Конец merge_pass_global

static float run_sort_once(const std::vector<float>& h, int block, size_t shared_limit_bytes){ // Один прогон сортировки
    int n = (int)h.size();                                            // Размер массива
    float *d_a=nullptr, *d_b=nullptr;                                 // Буферы на GPU
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));                    // Выделяем d_a
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(float)));                    // Выделяем d_b
    CUDA_CHECK(cudaMemcpy(d_a, h.data(), n*sizeof(float), cudaMemcpyHostToDevice)); // Копируем вход

    int tile = block * K;                                             // Размер тайла
    int grid_tiles = (n + tile - 1) / tile;                           // Количество тайл-блоков

    GpuTimer t;                                                       // Таймер
    float ms_total = 0.f;                                             // Накопитель времени

    size_t sh1 = (size_t)(2 * tile) * sizeof(float);                  // Shared для этапа 1
    t.tic();                                                          // Старт замера
    sort_tile_local_then_merge_shared<<<grid_tiles, block, sh1>>>(d_a, n); // Запуск этапа 1
    CUDA_CHECK(cudaGetLastError());                                   // Проверка ошибок
    CUDA_CHECK(cudaDeviceSynchronize());                              // Синхронизация
    ms_total += t.toc();                                              // Добавляем время

    int run = tile;                                                   // Текущий размер run
    bool ping = true;                                                 // Переключатель вход/выход

    while(run < n){                                                   // Пока run меньше n
        int pairs = (n + 2*run - 1) / (2*run);                         // Количество пар
        size_t sh = (size_t)(2 * run) * sizeof(float);                // Требуемый shared

        t.tic();                                                      // Старт замера merge-прохода
        if(sh <= shared_limit_bytes){                                 // Если хватает shared
            merge_pass_shared<<<pairs, 256, sh>>>(ping ? d_a : d_b, ping ? d_b : d_a, n, run); // Shared merge
        } else {                                                      // Иначе shared не хватает
            merge_pass_global<<<pairs, 256>>>(ping ? d_a : d_b, ping ? d_b : d_a, n, run);     // Global merge
        }                                                             // Конец if
        CUDA_CHECK(cudaGetLastError());                                // Проверка ошибок
        CUDA_CHECK(cudaDeviceSynchronize());                           // Синхронизация
        ms_total += t.toc();                                           // Добавляем время

        ping = !ping;                                                 // Меняем буферы
        run *= 2;                                                     // Удваиваем run
    }                                                                 // Конец while

    cudaFree(d_a);                                                    // Освобождаем d_a
    cudaFree(d_b);                                                    // Освобождаем d_b
    return ms_total;                                                  // Возвращаем общее время
}                                                                    // Конец run_sort_once

// =====================================================             // Разделитель
// Генерация данных                                                  // Раздел генерации
// =====================================================             // Разделитель
static std::vector<float> generate_data(int n, int seed){            // Генерация данных с seed
    std::mt19937 rng(seed);                                          // RNG
    std::uniform_real_distribution<float> dist(0.f, 1.f);            // Распределение [0,1]
    std::vector<float> h(n);                                         // Вектор размера n
    for(int i=0;i<n;i++) h[i] = dist(rng);                           // Заполнение
    return h;                                                        // Возвращаем
}                                                                    // Конец generate_data

// =====================================================             // Разделитель
// Простой SVG-график (без внешних библиотек)                         // Раздел SVG
// =====================================================             // Разделитель
static void write_svg_chart(                                         // Функция записи SVG
    const std::string& path,                                         // Путь к файлу SVG
    const std::string& title,                                        // Заголовок графика
    const std::string& x_label,                                      // Подпись оси X
    const std::string& y_label,                                      // Подпись оси Y
    const std::vector<int>& xs,                                      // Значения X
    const std::vector<std::pair<std::string, std::vector<float>>>& series // Серии данных
){                                                                    // Начало тела функции
    const int W = 1100, H = 650;                                     // Размер SVG
    const int L = 90, R = 30, T = 60, B = 80;                        // Отступы
    const int PW = W - L - R;                                        // Ширина области графика
    const int PH = H - T - B;                                        // Высота области графика

    float y_max = 0.f;                                               // Максимум по Y
    for (auto& s : series)                                           // Проходим по сериям
        for (float v : s.second)                                     // Проходим по значениям серии
            y_max = std::max(y_max, v);                               // Обновляем максимум
    if (y_max <= 0.f) y_max = 1.f;                                   // Защита от нуля
    y_max *= 1.10f;                                                  // Запас 10%

    auto x_to_px = [&](int i){                                       // Лямбда: индекс -> координата X
        if (xs.size() == 1) return L + PW/2;                         // Если одна точка — по центру
        return L + (int)std::round((double)i * PW / (double)(xs.size()-1)); // Линейная шкала
    };                                                               // Конец x_to_px
    auto y_to_py = [&](float y){                                     // Лямбда: значение -> координата Y
        double t = (double)y / (double)y_max;                        // Нормализация
        return T + (int)std::round((1.0 - t) * PH);                  // Перевод в пиксели (сверху вниз)
    };                                                               // Конец y_to_py

    std::ofstream f(path);                                           // Открываем файл SVG
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";             // Заголовок XML
    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H << "\">\n"; // Открытие SVG

    f << "<rect x=\"0\" y=\"0\" width=\"" << W << "\" height=\"" << H << "\" fill=\"white\"/>\n"; // Белый фон

    f << "<text x=\"" << (W/2) << "\" y=\"30\" font-size=\"22\" text-anchor=\"middle\">" << title << "</text>\n"; // Заголовок

    f << "<line x1=\"" << L << "\" y1=\"" << (T+PH) << "\" x2=\"" << (L+PW) << "\" y2=\"" << (T+PH) << "\" stroke=\"black\"/>\n"; // Ось X
    f << "<line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\"" << (T+PH) << "\" stroke=\"black\"/>\n";             // Ось Y

    f << "<text x=\"" << (W/2) << "\" y=\"" << (H-25) << "\" font-size=\"18\" text-anchor=\"middle\">" << x_label << "</text>\n";     // Подпись X
    f << "<text x=\"25\" y=\"" << (H/2) << "\" font-size=\"18\" text-anchor=\"middle\" transform=\"rotate(-90 25," << (H/2) << ")\">" << y_label << "</text>\n"; // Подпись Y

    for(int k=0;k<=5;k++){                                           // 6 горизонтальных линий сетки
        float yv = y_max * k / 5.f;                                  // Значение деления Y
        int py = y_to_py(yv);                                        // Координата Y в пикселях
        f << "<line x1=\"" << L << "\" y1=\"" << py << "\" x2=\"" << (L+PW) << "\" y2=\"" << py << "\" stroke=\"#cccccc\"/>\n";       // Линия сетки
        f << "<text x=\"" << (L-10) << "\" y=\"" << (py+5) << "\" font-size=\"14\" text-anchor=\"end\">"
          << std::fixed << std::setprecision(2) << yv << "</text>\n"; // Подпись деления Y
    }                                                                // Конец делений Y

    for(size_t i=0;i<xs.size();i++){                                 // Метки X
        int px = x_to_px((int)i);                                    // Координата X в пикселях
        f << "<text x=\"" << px << "\" y=\"" << (T+PH+25) << "\" font-size=\"14\" text-anchor=\"middle\">" << xs[i] << "</text>\n";   // Подпись X
    }                                                                // Конец меток X

    std::vector<std::string> dashes = {"", "8,6", "2,6", "12,4,2,4"}; // Набор штриховок для различия серий

    int leg_x = L + 20, leg_y = T + 10;                              // Координаты легенды

    for(size_t si=0; si<series.size(); si++){                        // Цикл по сериям
        const auto& name = series[si].first;                         // Название серии
        const auto& ys   = series[si].second;                        // Данные серии

        std::ostringstream pts;                                      // Строка точек polyline
        for(size_t i=0;i<ys.size();i++){                             // Цикл по точкам серии
            int px = x_to_px((int)i);                                // X координата точки
            int py = y_to_py(ys[i]);                                 // Y координата точки
            pts << px << "," << py << " ";                           // Добавляем "x,y" в список точек
        }                                                            // Конец цикла точек

        f << "<polyline points=\"" << pts.str()                      // Пишем polyline с точками
          << "\" fill=\"none\" stroke=\"black\" stroke-width=\"2\""; // Стиль линии
        if(!dashes[si % dashes.size()].empty())                      // Если нужен dasharray
            f << " stroke-dasharray=\"" << dashes[si % dashes.size()] << "\""; // Указываем штрихи
        f << "/>\n";                                                 // Закрываем polyline

        for(size_t i=0;i<ys.size();i++){                             // Рисуем маркеры-точки
            int px = x_to_px((int)i);                                // X координата маркера
            int py = y_to_py(ys[i]);                                 // Y координата маркера
            f << "<circle cx=\"" << px << "\" cy=\"" << py << "\" r=\"4\" fill=\"black\"/>\n"; // Кружок
        }                                                            // Конец маркеров

        int ly = leg_y + (int)si * 22;                               // Y координата строки легенды
        f << "<line x1=\"" << leg_x << "\" y1=\"" << ly << "\" x2=\"" << (leg_x+40) << "\" y2=\"" << ly
          << "\" stroke=\"black\" stroke-width=\"2\"";               // Линия в легенде
        if(!dashes[si % dashes.size()].empty())                      // Если нужен dasharray
            f << " stroke-dasharray=\"" << dashes[si % dashes.size()] << "\""; // Штрихи в легенде
        f << "/>\n";                                                 // Закрываем линию легенды
        f << "<text x=\"" << (leg_x+55) << "\" y=\"" << (ly+5) << "\" font-size=\"14\">" << name << "</text>\n"; // Текст легенды
    }                                                                // Конец серий

    f << "</svg>\n";                                                 // Закрываем SVG
}                                                                    // Конец write_svg_chart


// =====================================================                        // Разделитель секции
// MAIN                                                     // Заголовок секции main
// =====================================================                        // Разделитель секции
int main(){                                                  // Точка входа в программу
    std::cout << "=============================================\n"; // Печать разделителя
    std::cout << "Задание 4: Замеры времени и CSV + SVG графики\n"; // Печать названия программы
    std::cout << "=============================================\n\n"; // Печать разделителя и пустой строки

    cudaDeviceProp prop{};                                   // Структура для свойств GPU
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));            // Получаем свойства устройства GPU #0
    size_t shared_limit = prop.sharedMemPerBlock;             // Сохраняем лимит shared memory на блок

    std::cout << "GPU: " << prop.name << "\n";                // Выводим имя GPU
    std::cout << "Лимит shared memory на блок: " << shared_limit << " байт\n\n"; // Выводим лимит shared memory

    std::vector<int> sizes = { 10000, 100000, 1000000 };      // Список размеров массивов для экспериментов

    const int iters_reduce = 30;                              // Количество повторов для редукции (усреднение)
    const int iters_sort   = 5;                               // Количество повторов для сортировки (усреднение)
    const int block_reduce = 256;                             // Размер блока для редукции
    const int block_sort   = 256;                             // Размер блока для сортировки (этап 1)

    make_results_dir();                                       // Создаём папку results (если её нет)

    std::ofstream csv("results/times.csv");                   // Открываем CSV файл для записи результатов
    if(!csv){                                                 // Проверяем, удалось ли открыть файл
        std::cerr << "Ошибка: не удалось создать файл results/times.csv\n"; // Сообщаем об ошибке
        return 1;                                             // Завершаем с кодом ошибки
    }                                                         // Конец проверки файла
    csv << "task,variant,n,ms\n";                              // Пишем заголовок CSV

// показ таблицы в консоль
    std::cout << std::left                                    // Включаем выравнивание влево
              << std::setw(10) << "Задача"                     // Колонка: задача
              << std::setw(18) << "Вариант"                    // Колонка: вариант
              << std::setw(12) << "n"                          // Колонка: размер n
              << std::setw(12) << "мс"                         // Колонка: время в мс
              << "\n";                                         // Перевод строки
    std::cout << std::string(52, '-') << "\n";                 // Печать разделительной линии

    // данные для графиков                                      // Комментарий: векторы для построения графиков
    std::vector<float> reduce_global_ms, reduce_shared_ms, sort_ms; // Храним измеренные времена по каждому n

    for(int n : sizes){                                       // Проходим по всем размерам массивов
        // ---- reduce ----                                     // Раздел: редукция
        auto h_reduce = generate_data(n, 42);                  // Генерируем данные для редукции (seed=42)

        float ms_g = 0.f;                                      // Накопитель времени global-редукции
        for(int i=0;i<iters_reduce;i++) ms_g += run_reduce_once(h_reduce, false, block_reduce); // Повторы global
        ms_g /= iters_reduce;                                  // Среднее время global

        float ms_s = 0.f;                                      // Накопитель времени shared-редукции
        for(int i=0;i<iters_reduce;i++) ms_s += run_reduce_once(h_reduce, true, block_reduce);  // Повторы shared
        ms_s /= iters_reduce;                                  // Среднее время shared

        reduce_global_ms.push_back(ms_g);                      // Добавляем global время для графика
        reduce_shared_ms.push_back(ms_s);                      // Добавляем shared время для графика

        std::cout << std::left                                 // Выравнивание вывода влево
                  << std::setw(10) << "reduce"                 // Строка: задача reduce
                  << std::setw(18) << "global"                 // Вариант global
                  << std::setw(12) << n                        // Размер n
                  << std::setw(12) << std::fixed << std::setprecision(4) << ms_g << "\n"; // Время global

        std::cout << std::left                                 // Выравнивание вывода влево
                  << std::setw(10) << "reduce"                 // Строка: задача reduce
                  << std::setw(18) << "shared"                 // Вариант shared
                  << std::setw(12) << n                        // Размер n
                  << std::setw(12) << std::fixed << std::setprecision(4) << ms_s << "\n"; // Время shared

        csv << "reduce,global," << n << "," << ms_g << "\n";    // Записываем в CSV результат global-редукции
        csv << "reduce,shared," << n << "," << ms_s << "\n";    // Записываем в CSV результат shared-редукции

        // ---- sort ----                                       // Раздел: сортировка
        auto h_sort = generate_data(n, 123);                    // Генерируем данные для сортировки (seed=123)

        float ms_sort = 0.f;                                    // Накопитель времени сортировки
        for(int i=0;i<iters_sort;i++) ms_sort += run_sort_once(h_sort, block_sort, shared_limit); // Повторы сортировки
        ms_sort /= iters_sort;                                  // Среднее время сортировки

        sort_ms.push_back(ms_sort);                             // Добавляем время сортировки для графика

        std::cout << std::left                                  // Выравнивание вывода влево
                  << std::setw(10) << "sort"                    // Строка: задача sort
                  << std::setw(18) << "local+merge"             // Вариант local+merge
                  << std::setw(12) << n                         // Размер n
                  << std::setw(12) << std::fixed << std::setprecision(4) << ms_sort << "\n\n"; // Время sort + пустая строка

        csv << "sort,local_sharedmerge," << n << "," << ms_sort << "\n"; // Записываем в CSV результат сортировки
    }                                                           // Конец цикла по sizes

    csv.close();                                                 // Закрываем CSV файл

    // SVG-графики                                               // Комментарий: строим SVG графики
    write_svg_chart(                                             // Запись SVG для редукции
        "results/reduce_times.svg",                              // Путь к SVG редукции
        "Редукция суммы: время vs размер массива",               // Заголовок графика редукции
        "Размер массива (n)",                                    // Подпись оси X
        "Время (мс)",                                            // Подпись оси Y
        sizes,                                                   // Значения X
        {                                                       // Список серий
            {"reduce_global", reduce_global_ms},                 // Серия: global
            {"reduce_shared", reduce_shared_ms}                  // Серия: shared
        }                                                       // Конец списка серий
    );                                                          // Конец вызова write_svg_chart (reduce)

    write_svg_chart(                                             // Запись SVG для сортировки
        "results/sort_times.svg",                                // Путь к SVG сортировки
        "Сортировка: время vs размер массива",                   // Заголовок графика сортировки
        "Размер массива (n)",                                    // Подпись оси X
        "Время (мс)",                                            // Подпись оси Y
        sizes,                                                   // Значения X
        {                                                       // Список серий
            {"sort_local+merge", sort_ms}                        // Серия: сортировка
        }                                                       // Конец списка серий
    );                                                          // Конец вызова write_svg_chart (sort)

    std::cout << "Сохранено:\n";                                 // Сообщение: файлы сохранены
    std::cout << "  1) results/times.csv\n";                     // Перечень: CSV
    std::cout << "  2) results/reduce_times.svg\n";              // Перечень: reduce SVG
    std::cout << "  3) results/sort_times.svg\n\n";              // Перечень: sort SVG + пустая строка
    std::cout << "Открой SVG двойным кликом (браузер/VS Code) — это готовые графики.\n"; // Подсказка открытия SVG
    std::cout << "Готово.\n";                                    // Сообщение о завершении

    return 0;                                                   // Успешное завершение программы
}                                                               // Конец main
