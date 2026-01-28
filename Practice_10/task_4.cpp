// mpi_agg_scaling.cpp                                                     // Имя файла: MPI агрегирование и масштабирование
// MPI strong/weak scaling: aggregate over large array (sum/min/max)        // Strong/weak scaling: агрегируем большой массив (sum/min/max)
// Measures compute, communication (Reduce/Allreduce), total time.          // Измеряем compute, коммуникацию (Reduce/Allreduce) и total time
// Build:  mpicxx -O3 mpi_agg_scaling.cpp -o mpi_agg                        // Сборка: компиляция mpicxx с -O3
// Run:    mpirun -np 4 ./mpi_agg 100000000 reduce                          // Пример запуска: 4 процесса, N=1e8, режим reduce
//         mpirun -np 4 ./mpi_agg 100000000 allreduce                       // Пример запуска: 4 процесса, N=1e8, режим allreduce
// Args:   N_global (for strong scaling), mode = reduce|allreduce           // Аргументы: N_global (для strong), mode=reduce|allreduce
// For weak scaling: pass N_local_per_rank and add flag "weak" (see below)  // Для weak: передай N_local_per_rank и флаг "weak" третьим аргументом

#include <mpi.h>                                                          // MPI API (MPI_Init, MPI_Reduce, MPI_Allreduce, MPI_Wtime, ...)
#include <iostream>                                                       // std::cout / std::cerr
#include <vector>                                                         // std::vector для локального массива
#include <algorithm>                                                      // std::min / std::max
#include <numeric>                                                        // (необязательно здесь) численные алгоритмы
#include <cstdlib>                                                        // std::atoll и т.п.
#include <iomanip>                                                        // форматированный вывод (fixed, setprecision)
#include <cmath>                                                          // математика (может понадобиться, здесь почти не используется)
#include <string>                                                         // std::string
                                                                           
struct Agg {                                                               // Структура для агрегатов: сумма, минимум, максимум
    double sum;                                                            // Сумма элементов
    double mn;                                                             // Минимум
    double mx;                                                             // Максимум
};                                                                         // Конец struct Agg

static inline Agg local_aggregate(const std::vector<float>& a) {           // Функция: локально считаем sum/min/max на текущем ранге
    Agg r{};                                                               // Создаём результат (value-init)
    r.sum = 0.0;                                                           // Инициализация суммы
    r.mn  = std::numeric_limits<double>::infinity();                       // Инициализация минимума как +inf
    r.mx  = -std::numeric_limits<double>::infinity();                      // Инициализация максимума как -inf
    for (float x : a) {                                                    // Проходим по всем элементам локального массива
        r.sum += x;                                                        // Добавляем элемент к сумме
        r.mn = std::min(r.mn, (double)x);                                  // Обновляем минимум
        r.mx = std::max(r.mx, (double)x);                                  // Обновляем максимум
    }                                                                      // Конец цикла for
    return r;                                                              // Возвращаем локальные агрегаты
}                                                                          // Конец local_aggregate

int main(int argc, char** argv) {                                          // Точка входа программы
    MPI_Init(&argc, &argv);                                                // Инициализация MPI среды

    int rank=0, size=1;                                                    // rank: номер процесса, size: число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                  // Получаем rank текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);                                  // Получаем общее число процессов

    // Defaults                                                            // Значения по умолчанию
    long long N = (argc > 1) ? std::atoll(argv[1]) : 100000000LL; // 1e8    // N: либо из argv[1], либо 1e8
    std::string mode = (argc > 2) ? argv[2] : "reduce";          // reduce | allreduce // mode: reduce/allreduce
    std::string scaling = (argc > 3) ? argv[3] : "strong";       // strong | weak      // scaling: strong/weak

    bool use_allreduce = (mode == "allreduce");                              // Флаг: использовать Allreduce?
    bool weak = (scaling == "weak");                                         // Флаг: weak scaling?

    // For strong scaling: N = global size fixed.                             // Strong: N считается глобальным и фиксированным
    // For weak scaling:  N = local size per rank; global size = N * size.    // Weak: N считается локальным на ранг; глобальный = N * size
    long long N_global = weak ? (N * (long long)size) : N;                   // Вычисляем глобальный размер массива
    long long N_local  = weak ? N : (N_global / size);                       // Вычисляем локальный размер (по умолчанию равномерно)
    long long rem = weak ? 0 : (N_global % size);                            // Остаток при делении (только для strong)
    if (!weak) {                                                             // Если strong scaling
        // Distribute remainder to last rank (simple)                         // Простое распределение остатка: отдаём последнему рангу
        if (rank == size - 1) N_local += rem;                                // Последний rank получает +rem элементов
    }                                                                        // Конец if (!weak)

    // Create local chunk                                                     // Создаём локальный массив (чанк) на каждом ранге
    std::vector<float> a((size_t)N_local);                                   // Локальный массив длины N_local

    // Fill deterministically without communication                           // Заполняем детерминированно без коммуникаций
    // Each rank uses different seed pattern to avoid identical arrays.       // Каждый rank использует иной паттерн, чтобы массивы не совпадали
    for (long long i = 0; i < N_local; ++i) {                                // Цикл по локальным индексам
        // lightweight pseudo-random-ish pattern                              // Лёгкий псевдослучайный паттерн (без RNG и общения)
        unsigned int v = (unsigned int)( (rank + 1) * 2654435761u + (unsigned int)i ); // Генерируем значение на основе rank и i
        a[(size_t)i] = (float)((v % 1000u) * 0.001f); // [0, 0.999]          // Маппим в диапазон [0, 0.999]
    }                                                                        // Конец цикла заполнения

    // Warm-up barrier                                                        // Барьер-прогрев: выравниваем старт перед таймингом
    MPI_Barrier(MPI_COMM_WORLD);                                             // Все процессы доходят до этой точки

    // ---- Timing ----                                                       // Начинаем замеры времени
    double t0 = MPI_Wtime();                                                 // t0: старт общего времени

    // Compute local aggregate                                                // Вычисление локальных агрегатов (compute часть)
    double t_comp0 = MPI_Wtime();                                            // Старт compute тайминга
    Agg loc = local_aggregate(a);                                            // Считаем локально sum/min/max
    double t_comp1 = MPI_Wtime();                                            // Конец compute тайминга

    // Communication: sum/min/max                                             // Коммуникация: собираем sum/min/max по всем процессам
    Agg glob{};                                                              // Структура для глобальных результатов
    double t_comm0 = MPI_Wtime();                                            // Старт коммуникации

    if (use_allreduce) {                                                     // Если выбран allreduce
        MPI_Allreduce(&loc.sum, &glob.sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Allreduce суммы: результат у всех
        MPI_Allreduce(&loc.mn,  &glob.mn,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); // Allreduce минимума
        MPI_Allreduce(&loc.mx,  &glob.mx,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Allreduce максимума
    } else {                                                                 // Иначе используем reduce
        MPI_Reduce(&loc.sum, &glob.sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce суммы на rank 0
        MPI_Reduce(&loc.mn,  &glob.mn,  1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); // Reduce минимума на rank 0
        MPI_Reduce(&loc.mx,  &glob.mx,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Reduce максимума на rank 0
    }                                                                        // Конец if/else

    double t_comm1 = MPI_Wtime();                                            // Конец коммуникации

    // Ensure all ranks finish together before total stop                     // Выравниваем завершение, чтобы total время отражало "самый медленный" rank
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед остановкой общего таймера
    double t1 = MPI_Wtime();                                                 // t1: конец общего времени

    double comp = (t_comp1 - t_comp0);                                       // Время compute (локально на rank)
    double comm = (t_comm1 - t_comm0);                                       // Время коммуникации (локально на rank)
    double total = (t1 - t0);                                                // Общее время (compute + comm + барьеры/оверход)

    // Collect max times across ranks (critical for scaling)                  // Берём max по rank'ам (важно для анализа масштабирования)
    double comp_max=0, comm_max=0, total_max=0;                              // Переменные для максимальных времен по всем rank'ам
    MPI_Allreduce(&comp,  &comp_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Max compute time среди rank'ов
    MPI_Allreduce(&comm,  &comm_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Max comm time среди rank'ов
    MPI_Allreduce(&total, &total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Max total time среди rank'ов

    if (rank == 0) {                                                         // Выводим результаты только на rank 0
        std::cout << std::fixed << std::setprecision(6);                     // Формат: 6 знаков после запятой
        std::cout << "MPI agg (" << (weak ? "weak" : "strong") << " scaling), mode=" << mode << "\n"; // Заголовок: weak/strong и режим
        std::cout << "Processes P=" << size << "\n";                         // Число процессов P
        std::cout << "N_global=" << N_global << "  N_local~=" << (N_global / size) << "\n\n"; // Печать N_global и среднего N_local

        std::cout << "Max times across ranks (s):\n";                        // Заголовок: максимальные времена по всем rank'ам
        std::cout << "  compute_max : " << comp_max  << "\n";                // Печать compute_max
        std::cout << "  comm_max    : " << comm_max  << "\n";                // Печать comm_max
        std::cout << "  total_max   : " << total_max << "\n\n";              // Печать total_max

        if (!use_allreduce) {                                                // Если reduce
            std::cout << "Result available on rank 0 only (Reduce).\n";      // Результат есть только на rank 0
            std::cout << "sum=" << glob.sum << " min=" << glob.mn << " max=" << glob.mx << "\n"; // Печать глобальных агрегатов
        } else {                                                             // Если allreduce
            std::cout << "Result available on all ranks (Allreduce).\n";     // Результат доступен на всех rank'ах
            std::cout << "sum=" << glob.sum << " min=" << glob.mn << " max=" << glob.mx << "\n"; // Печать глобальных агрегатов
        }                                                                    // Конец if/else
    }                                                                        // Конец if rank==0

    MPI_Finalize();                                                          // Завершаем MPI среду
    return 0;                                                                // Завершаем программу успешно
}                                                                            // Конец main
