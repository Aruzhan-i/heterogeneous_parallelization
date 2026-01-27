// task2_gauss_mpi_timing.cpp                                               // Имя файла
// Practice 9 - Task 2: Distributed Gaussian Elimination (MPI) + TIMING      // Описание задания
// - rank 0 creates A (N x N) and b (N), saves copies A0/b0 for residual check // Rank 0 генерирует матрицу A и вектор b и сохраняет копии для проверки
// - rows are distributed across processes (block rows) via MPI_Scatterv (works for any N and any P) // Распределение строк между процессами через Scatterv
// - Forward elimination:                                                    // Прямой ход Гаусса
//     for k=0..N-1: owner of pivot row broadcasts pivot row using MPI_Bcast // Владелец опорной строки рассылает её всем через Bcast
//     each rank eliminates below pivot for its local rows                   // Каждый процесс зануляет элементы ниже опорного в своих строках
// - Gather triangular A and modified b to rank 0 via MPI_Gatherv            // Сбор верхнетреугольной матрицы и b на rank 0
// - Back substitution on rank 0                                             // Обратный ход на rank 0
// - Residual check on rank 0: ||A0 x - b0||_2 and max|A0 x - b0|            // Проверка невязки: L2-норма и max-отклонение
// - TIMING: Scatterv, ForwardElim, Gatherv, BackSub, Residual, Total        // Замеры времени по фазам
//   Times are reported as MAX over ranks for parallel phases.               // Для параллельных фаз берётся максимум по процессам
//                                                                           // Пустая строка-разделитель
// Build (MS-MPI + MinGW g++):                                               // Инструкция компиляции для Windows
//   g++ -O2 task2_gauss_mpi_timing.cpp -o task2.exe ^                       // Компиляция с оптимизацией -O2
//     -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^               // Путь к заголовкам MPI
//     -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi         // Путь к библиотеке и линковка msmpi
//                                                                           // Пустая строка
// Run:                                                                      // Инструкция запуска
//   "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 task2.exe 10       // Пример запуска на 4 процессах с N=10
//                                                                           // Пустая строка

#include <mpi.h>                                                            // Подключение MPI библиотеки
#include <iostream>                                                         // Для ввода/вывода
#include <vector>                                                           // Для std::vector
#include <random>                                                           // Для генерации случайных чисел
#include <cmath>                                                            // Для мат. функций: fabs, sqrt
#include <cstdlib>                                                          // Для atoi, exit
#include <iomanip>                                                          // Для setprecision, fixed
#include <algorithm>                                                        // Для std::max

static inline int owner_of_row(int row,                                     // Функция: найти владельца глобальной строки row
                               const std::vector<int>& row_counts,          // Кол-во строк у каждого процесса
                               const std::vector<int>& row_displs) {        // Смещения начала блока строк каждого процесса
    int size = (int)row_counts.size();                                       // Общее число процессов (по длине массива counts)
    for (int r = 0; r < size; ++r) {                                         // Перебор всех процессов
        int start = row_displs[r];                                           // Первая строка, принадлежащая процессу r
        int end   = start + row_counts[r];                                   // Последняя строка (не включительно) у r
        if (row >= start && row < end) return r;                             // Если row попадает в диапазон — владелец найден
    }                                                                        // Конец цикла по процессам
    return 0;                                                                // Фоллбек: если не нашли (не должно быть), вернуть 0
}                                                                            // Конец функции owner_of_row

int main(int argc, char** argv) {                                            // Главная функция программы
    MPI_Init(&argc, &argv);                                                  // Инициализация MPI окружения

    int rank = 0, size = 1;                                                  // Переменные: номер процесса и число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                    // Получаем rank текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);                                    // Получаем общее число процессов size

    // -------------------- Parse N --------------------                      // Раздел: чтение размера системы N
    int N = 8;                                                               // Значение N по умолчанию
    if (argc >= 2) {                                                         // Если передан аргумент командной строки
        N = std::atoi(argv[1]);                                              // Преобразуем аргумент в int
        if (N < 1) N = 1;                                                    // Защита: N не может быть меньше 1
    }                                                                        // Конец обработки аргументов

    // -------------------- Build row distribution --------------------       // Раздел: построение распределения строк по процессам
    std::vector<int> row_counts(size, 0);                                    // Количество строк на каждый процесс
    std::vector<int> row_displs(size, 0);                                    // Смещения начала строковых блоков

    int base = N / size;                                                     // Базовое число строк на процесс
    int rem  = N % size;                                                     // Остаток строк, которые нужно распределить

    for (int r = 0; r < size; ++r) {                                         // Цикл по процессам
        row_counts[r] = base + (r < rem ? 1 : 0);                             // Первые rem процессов получают +1 строку
    }                                                                        // Конец цикла распределения row_counts
    row_displs[0] = 0;                                                       // Смещение для первого процесса = 0
    for (int r = 1; r < size; ++r) {                                         // Вычисление смещений для остальных процессов
        row_displs[r] = row_displs[r - 1] + row_counts[r - 1];               // Суммируем строки предыдущих процессов
    }                                                                        // Конец цикла по displs

    int local_rows   = row_counts[rank];                                     // Число строк у текущего процесса
    int local_A_elems = local_rows * N;                                      // Кол-во элементов матрицы A на процессе

    std::vector<double> A_local(local_A_elems, 0.0);                         // Локальная часть матрицы A (строки блока)
    std::vector<double> b_local(local_rows, 0.0);                            // Локальная часть вектора b

    // -------------------- Rank 0 creates A, b and saves A0, b0 -------------------- // Раздел: создание исходных данных на rank 0
    std::vector<double> A, b, A0, b0;                                        // Глобальные матрица/вектор и их копии

    // Timing: generation on rank 0 only                                     // Замер времени генерации только на rank 0
    double gen_time = 0.0;                                                   // Время генерации
    if (rank == 0) {                                                         // Только процесс 0 генерирует данные
        double t0 = MPI_Wtime();                                             // Начало таймера генерации

        A.resize((size_t)N * (size_t)N);                                     // Выделяем память под матрицу N×N
        b.resize((size_t)N);                                                 // Выделяем память под вектор b размера N

        std::mt19937_64 rng(12345);                                          // Инициализация генератора случайных чисел
        std::uniform_real_distribution<double> dist(-1.0, 1.0);              // Равномерное распределение [-1, 1]

        for (int i = 0; i < N; ++i) {                                        // Цикл по строкам
            double rowsum = 0.0;                                             // Сумма модулей в строке для диагонального доминирования
            for (int j = 0; j < N; ++j) {                                    // Цикл по столбцам
                double v = dist(rng);                                        // Генерация случайного элемента
                A[(size_t)i * N + j] = v;                                    // Запись элемента в A
                rowsum += std::fabs(v);                                      // Добавляем |v| к rowsum
            }                                                                // Конец цикла по столбцам
            A[(size_t)i * N + i] += (double)N + rowsum;                      // Усиливаем диагональ для устойчивости
            b[(size_t)i] = dist(rng);                                        // Заполняем b случайным значением
        }                                                                    // Конец цикла по строкам

        A0 = A;                                                              // Сохраняем копию исходной матрицы A
        b0 = b;                                                              // Сохраняем копию исходного вектора b

        double t1 = MPI_Wtime();                                             // Конец таймера генерации
        gen_time = t1 - t0;                                                  // Длительность генерации
    }                                                                        // Конец if rank==0

    // -------------------- Scatterv meta for A --------------------          // Подготовка counts/displs для Scatterv матрицы A
    std::vector<int> A_counts(size, 0);                                      // Кол-во элементов A для отправки каждому процессу
    std::vector<int> A_displs(size, 0);                                      // Смещения элементов A для каждого процесса
    for (int r = 0; r < size; ++r) {                                         // Цикл по процессам
        A_counts[r] = row_counts[r] * N;                                     // Элементов = строк * N
        A_displs[r] = row_displs[r] * N;                                     // Смещение по элементам = смещение строк * N
    }                                                                        // Конец цикла построения A_counts/A_displs

    const double EPS = 1e-12;                                                // Порог для проверки нулевого пивота

    // -------------------- Timing variables --------------------             // Переменные для измерения времени
    double t0_total = 0.0, t1_total = 0.0;                                   // Начало/конец общего таймера

    double t_sc0 = 0.0, t_sc1 = 0.0;                                         // Время Scatterv: start/end
    double t_fw0 = 0.0, t_fw1 = 0.0;                                         // Время forward elimination: start/end
    double t_ga0 = 0.0, t_ga1 = 0.0;                                         // Время Gatherv: start/end

    // BackSub + Residual only on rank 0, but we'll time them separately      // BackSub и Residual только на rank0
    double back_time = 0.0;                                                  // Время обратного хода
    double resid_time = 0.0;                                                 // Время вычисления невязки

    MPI_Barrier(MPI_COMM_WORLD);                                             // Синхронизация всех процессов
    t0_total = MPI_Wtime();                                                  // Старт общего таймера

    // -------------------- Scatter timing --------------------               // Тайминг Scatterv
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед Scatterv для честного измерения
    t_sc0 = MPI_Wtime();                                                     // Start Scatterv timer

    MPI_Scatterv(                                                            // Рассылка строк матрицы A процессам
        rank == 0 ? A.data() : nullptr,                                      // Отправляющий буфер только у rank0
        A_counts.data(),                                                     // Кол-во элементов каждому процессу
        A_displs.data(),                                                     // Смещение каждого блока в A
        MPI_DOUBLE,                                                          // Тип данных: double
        A_local.data(),                                                      // Приёмный буфер локальной части A
        local_A_elems,                                                       // Кол-во элементов, получаемых текущим процессом
        MPI_DOUBLE,                                                          // Тип данных: double
        0,                                                                   // Root процесса-рассыльщика
        MPI_COMM_WORLD                                                       // Коммуникатор
    );                                                                       // Конец Scatterv A

    MPI_Scatterv(                                                            // Рассылка соответствующих строк b процессам
        rank == 0 ? b.data() : nullptr,                                      // Отправляющий буфер b только у rank0
        row_counts.data(),                                                   // Сколько элементов b каждому процессу (по числу строк)
        row_displs.data(),                                                   // Смещения по b
        MPI_DOUBLE,                                                          // Тип данных
        b_local.data(),                                                      // Приёмный буфер b_local
        local_rows,                                                          // Кол-во элементов b на текущем процессе
        MPI_DOUBLE,                                                          // Тип данных
        0,                                                                   // Root
        MPI_COMM_WORLD                                                       // Коммуникатор
    );                                                                       // Конец Scatterv b

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер: гарантируем завершение Scatterv
    t_sc1 = MPI_Wtime();                                                     // End Scatterv timer

    // -------------------- Forward elimination timing --------------------   // Тайминг прямого хода
    std::vector<double> pivot_row(N, 0.0);                                   // Буфер для опорной строки
    double pivot_b = 0.0;                                                    // Опорный элемент b[k] для текущего k

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед прямым ходом
    t_fw0 = MPI_Wtime();                                                     // Start forward elimination timer

    for (int k = 0; k < N; ++k) {                                            // Основной цикл по опорным строкам
        int owner = owner_of_row(k, row_counts, row_displs);                 // Определяем владельца строки k

        if (rank == owner) {                                                 // Если текущий процесс владелец pivot
            int local_k = k - row_displs[rank];                              // Локальный индекс pivot строки
            for (int j = 0; j < N; ++j) {                                    // Копируем pivot строку в pivot_row
                pivot_row[j] = A_local[(size_t)local_k * N + j];             // Запись элемента pivot строки
            }                                                                // Конец копирования pivot_row
            pivot_b = b_local[local_k];                                      // Pivot элемент вектора b
        }                                                                    // Конец if owner

        MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);   // Рассылаем pivot строку всем процессам
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);           // Рассылаем pivot_b всем процессам

        double pivot = pivot_row[k];                                         // Значение опорного элемента pivot
        if (std::fabs(pivot) < EPS) {                                        // Проверка на слишком маленький pivot
            if (rank == 0) {                                                 // Сообщение выводит только rank 0
                std::cerr << "Zero/near-zero pivot at k=" << k               // Печать ошибки pivot
                          << ". (No partial pivoting in this version.)\n";   // Уточнение что pivoting не реализован
            }                                                                // Конец if rank==0
            MPI_Abort(MPI_COMM_WORLD, 1);                                    // Аварийное завершение MPI программы
        }                                                                    // Конец проверки pivot

        for (int li = 0; li < local_rows; ++li) {                            // Перебор локальных строк процесса
            int gi = row_displs[rank] + li;                                  // Глобальный индекс строки
            if (gi <= k) continue;                                           // Пропускаем строки выше/на pivot

            double a_ik = A_local[(size_t)li * N + k];                       // Элемент A[i,k] для зануления
            double factor = a_ik / pivot;                                    // Коэффициент исключения

            A_local[(size_t)li * N + k] = 0.0;                               // Зануляем элемент под диагональю
            for (int j = k + 1; j < N; ++j) {                                // Обновление оставшейся части строки
                A_local[(size_t)li * N + j] -= factor * pivot_row[j];        // A[i,j] = A[i,j] - factor*pivot_row[j]
            }                                                                // Конец обновления строки A
            b_local[li] -= factor * pivot_b;                                 // b[i] = b[i] - factor * b_pivot
        }                                                                    // Конец цикла по локальным строкам
    }                                                                        // Конец цикла по pivot k

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер: все завершили forward elimination
    t_fw1 = MPI_Wtime();                                                     // End forward elimination timer

    // -------------------- Gather timing --------------------                // Тайминг сборки результатов
    if (rank == 0) {                                                         // Только root готовит буферы для сборки
        A.assign((size_t)N * (size_t)N, 0.0);                                // Выделение/обнуление A
        b.assign((size_t)N, 0.0);                                            // Выделение/обнуление b
    }                                                                        // Конец if rank==0

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед Gatherv
    t_ga0 = MPI_Wtime();                                                     // Start gather timer

    MPI_Gatherv(                                                             // Сбор локальных частей A в глобальную A на rank0
        A_local.data(),                                                      // Отправляемые данные A_local
        local_A_elems,                                                       // Кол-во отправляемых элементов
        MPI_DOUBLE,                                                          // Тип данных
        rank == 0 ? A.data() : nullptr,                                      // Приёмный буфер только у root
        A_counts.data(),                                                     // Сколько элементов принять от каждого процесса
        A_displs.data(),                                                     // Смещения куда класть данные
        MPI_DOUBLE,                                                          // Тип данных
        0,                                                                   // Root
        MPI_COMM_WORLD                                                       // Коммуникатор
    );                                                                       // Конец Gatherv для A

    MPI_Gatherv(                                                             // Сбор локальных частей b на rank0
        b_local.data(),                                                      // Отправляем b_local
        local_rows,                                                          // Количество элементов b_local
        MPI_DOUBLE,                                                          // Тип
        rank == 0 ? b.data() : nullptr,                                      // Приём только у root
        row_counts.data(),                                                   // Кол-во элементов от каждого процесса
        row_displs.data(),                                                   // Смещения
        MPI_DOUBLE,                                                          // Тип
        0,                                                                   // Root
        MPI_COMM_WORLD                                                       // Коммуникатор
    );                                                                       // Конец Gatherv b

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер после сборки
    t_ga1 = MPI_Wtime();                                                     // End gather timer

    // -------------------- Total end (parallel part finished) -------------------- // Конец измерения параллельной части
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер: параллельная часть полностью закончена
    t1_total = MPI_Wtime();                                                  // End total timer

    // -------------------- Reduce MAX timings across ranks -------------------- // Вычисляем максимум времени по процессам
    double scatter_t = t_sc1 - t_sc0;                                        // Scatter time на данном rank
    double forward_t = t_fw1 - t_fw0;                                        // Forward time на данном rank
    double gather_t  = t_ga1 - t_ga0;                                        // Gather time на данном rank
    double total_t   = t1_total - t0_total;                                  // Total parallel time на данном rank

    double scatter_max = 0.0, forward_max = 0.0, gather_max = 0.0, total_max = 0.0; // Максимальные времена (на root)

    MPI_Reduce(&scatter_t, &scatter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // MAX Scatter time
    MPI_Reduce(&forward_t, &forward_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // MAX Forward time
    MPI_Reduce(&gather_t,  &gather_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // MAX Gather time
    MPI_Reduce(&total_t,   &total_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // MAX Total time

    // -------------------- Back substitution + residual (rank 0 only) -------------------- // Обратный ход + проверка только на root
    std::vector<double> x;                                                   // Вектор решения x

    if (rank == 0) {                                                         // Только rank 0 выполняет back-substitution
        x.assign(N, 0.0);                                                    // Инициализируем решение нулями

        double t0 = MPI_Wtime();                                             // Start timer back-substitution

        for (int i = N - 1; i >= 0; --i) {                                   // Обратный ход: от последней строки к первой
            double diag = A[(size_t)i * N + i];                              // Диагональный элемент A[i,i]
            if (std::fabs(diag) < EPS) {                                     // Проверка на вырожденность диагонали
                std::cerr << "Zero/near-zero diagonal at i=" << i << "\n";   // Вывод ошибки
                MPI_Abort(MPI_COMM_WORLD, 2);                                // Прерывание выполнения
            }                                                                // Конец проверки диагонали

            double s = b[(size_t)i];                                         // Начинаем с правой части b[i]
            for (int j = i + 1; j < N; ++j) {                                // Вычитаем известные компоненты
                s -= A[(size_t)i * N + j] * x[j];                            // s = s - A[i,j]*x[j]
            }                                                                // Конец цикла по j
            x[i] = s / diag;                                                 // Находим x[i]
        }                                                                    // Конец обратного хода

        double t1 = MPI_Wtime();                                             // End back-substitution timer
        back_time = t1 - t0;                                                 // Записываем back_time

        // Residual check                                                     // Проверка невязки
        double t2 = MPI_Wtime();                                             // Start residual timer

        double norm2 = 0.0;                                                  // Накопитель для ||r||_2
        double max_abs = 0.0;                                                // Максимум |r_i|

        for (int i = 0; i < N; ++i) {                                        // Перебор строк для r = A0*x - b0
            double s = 0.0;                                                  // Аккумулятор для A0[i,:]*x
            for (int j = 0; j < N; ++j) {                                    // Умножаем строку i матрицы A0 на x
                s += A0[(size_t)i * N + j] * x[j];                           // s += A0[i,j]*x[j]
            }                                                                // Конец цикла по j
            double ri = s - b0[(size_t)i];                                   // r_i = (A0*x)_i - b0_i
            norm2 += ri * ri;                                                // Накапливаем сумму квадратов
            max_abs = std::max(max_abs, std::fabs(ri));                      // Обновляем максимум |r_i|
        }                                                                    // Конец цикла по i

        norm2 = std::sqrt(norm2);                                            // Итоговая L2-норма невязки

        double t3 = MPI_Wtime();                                             // End residual timer
        resid_time = t3 - t2;                                                // Записываем resid_time

        // Print solution                                                     // Вывод решения
        std::cout << "Solution x (N=" << N << ", P=" << size << "):\n";      // Заголовок с N и количеством процессов
        std::cout << std::fixed << std::setprecision(6);                     // Формат: фиксированный, 6 знаков после запятой

        const int K = 5; // how many elements to print from start/end         // Сколько элементов выводить в начале и конце

        if (N <= 2 * K) {                                                    // Если N маленькое — печатаем всё
            // If small system, print all                                     // Комментарий: печать полного решения
            for (int i = 0; i < N; ++i) {                                    // Цикл по всем элементам
                std::cout << "x[" << i << "] = " << x[i] << "\n";            // Печать x[i]
            }                                                                // Конец цикла
        } else {                                                             // Иначе печатаем часть
            // Print first K                                                  // Печать первых K
            for (int i = 0; i < K; ++i) {                                    // Цикл по первым K
                std::cout << "x[" << i << "] = " << x[i] << "\n";            // Вывод x[i]
            }                                                                // Конец цикла первых K

            std::cout << "...\n";                                            // Пропуск средних элементов

            // Print last K                                                   // Печать последних K
            for (int i = N - K; i < N; ++i) {                                // Цикл по последним K
                std::cout << "x[" << i << "] = " << x[i] << "\n";            // Вывод x[i]
            }                                                                // Конец цикла последних K
        }                                                                    // Конец if/else по размеру N



        // Print residual in scientific format                                // Печать невязки в scientific формате
        std::cout.setf(std::ios::scientific);                                // Переключаем cout в scientific
        std::cout << "Residual check:\n";                                    // Заголовок невязки
        std::cout << "  ||Ax - b||_2 = " << std::setprecision(6) << norm2 << "\n"; // Вывод L2-нормы
        std::cout << "  max|Ax - b|  = " << std::setprecision(6) << max_abs << "\n"; // Вывод max нормы
        std::cout.unsetf(std::ios::scientific);                              // Возвращаем обычный формат

        // Print timing summary                                               // Печать сводки времени
        std::cout << "Timing (max over ranks for parallel phases):\n";       // Заголовок тайминга
        std::cout << "  Gen (rank0 only): " << gen_time    * 1000.0 << " ms\n"; // Время генерации
        std::cout << "  Scatterv       : " << scatter_max * 1000.0 << " ms\n"; // Время Scatterv (max)
        std::cout << "  Forward elim   : " << forward_max * 1000.0 << " ms\n"; // Время прямого хода (max)
        std::cout << "  Gatherv        : " << gather_max  * 1000.0 << " ms\n"; // Время Gather (max)
        std::cout << "  BackSub (rank0): " << back_time   * 1000.0 << " ms\n"; // Время back-substitution
        std::cout << "  Resid (rank0)  : " << resid_time  * 1000.0 << " ms\n"; // Время residual
        std::cout << "  Total(par part): " << total_max   * 1000.0 << " ms\n"; // Общее параллельное время (max)
    }                                                                        // Конец if rank==0

    MPI_Finalize();                                                          // Завершение MPI
    return 0;                                                                // Успешное завершение программы
}                                                                            // Конец main
