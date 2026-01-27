// task3_floyd_warshall_mpi_timing.cpp                                       // Имя файла
// Practice 9 - Task 3: Parallel Floyd–Warshall (MPI) + TIMING + short output // Описание: параллельный Floyd–Warshall + тайминг + короткий вывод
//                                                                           // Пустая строка
// Adds:                                                                     // Что добавлено в этой версии
// - MPI_Wtime timers (Scatterv, InitAllgather, MainLoop, Print, Total)      // Таймеры MPI_Wtime по фазам
// - Prints "Execution time: <sec> seconds." on rank 0 (as required)         // Печатает требуемую строку времени выполнения на rank 0
// - Short output: full matrix only for small N, otherwise prints corners + checksum // Короткий вывод: либо вся матрица, либо углы + checksum
//                                                                           // Пустая строка

#include <mpi.h>                                                            // MPI функции и типы
#include <iostream>                                                         // std::cout / std::cerr
#include <vector>                                                           // std::vector
#include <random>                                                           // генераторы случайных чисел
#include <algorithm>                                                        // std::max
#include <iomanip>                                                          // std::setw, std::setprecision, std::fixed
#include <cstdlib>                                                          // std::atoi
#include <limits>                                                           // numeric_limits (здесь напрямую не используется, но подключено)
#include <cstdint>                                                          // std::uint64_t

static inline int owner_of_row(int row,                                     // Функция: определить, какой процесс владеет строкой row
                               const std::vector<int>& row_counts,          // Количество строк у каждого процесса
                               const std::vector<int>& row_displs) {        // Смещения (начальные индексы строк) для каждого процесса
    int size = (int)row_counts.size();                                       // Число процессов = размер массива row_counts
    for (int r = 0; r < size; ++r) {                                         // Перебираем все процессы
        int start = row_displs[r];                                           // Начальная строка, принадлежащая процессу r
        int end   = start + row_counts[r];                                   // Конечная граница (не включительно) строк процесса r
        if (row >= start && row < end) return r;                             // Если row попадает в этот диапазон — возвращаем владельца r
    }                                                                        // Конец цикла по процессам
    return 0;                                                                // Фоллбек (не должно происходить при корректных displs/counts)
}                                                                            // Конец owner_of_row

static inline void print_matrix_short(const std::vector<int>& dist_full,     // Функция: печать результата в сокращённом виде
                                      int N, int INF,                       // N = размер, INF = значение "бесконечности"
                                      int rank, int size) {                 // rank/size для вывода только на root и печати параметров
    if (rank != 0) return;                                                   // Печатает только процесс 0

    std::cout << "Floyd-Warshall result (N=" << N << ", P=" << size << "):\n"; // Заголовок вывода

    const int FULL_PRINT_MAX = 12; // print full matrix only if N <= 12      // Порог: печатать полностью только если N <= 12
    const int K = 5;              // otherwise print KxK corners             // Иначе печатать K×K углы матрицы

    auto cell = [&](int i, int j) -> int {                                   // Лямбда для доступа к элементу (i,j) в dist_full
        return dist_full[(size_t)i * (size_t)N + (size_t)j];                 // Индексация в 1D массиве (row-major)
    };                                                                       // Конец лямбды cell

    if (N <= FULL_PRINT_MAX) {                                               // Если матрица небольшая — печатаем целиком
        for (int i = 0; i < N; ++i) {                                        // Цикл по строкам
            for (int j = 0; j < N; ++j) {                                    // Цикл по столбцам
                int v = cell(i, j);                                          // Берём значение расстояния dist[i][j]
                if (v >= INF / 2) std::cout << std::setw(6) << "INF";        // Если расстояние очень большое — печатаем INF
                else              std::cout << std::setw(6) << v;            // Иначе печатаем само значение
            }                                                                // Конец цикла по столбцам
            std::cout << "\n";                                               // Переход на новую строку после строки матрицы
        }                                                                    // Конец цикла по строкам
        return;                                                              // Выходим после полного вывода
    }                                                                        // Конец if N<=FULL_PRINT_MAX

    // Print top-left corner                                                  // Комментарий: печать верхнего левого угла
    std::cout << "Top-left " << K << "x" << K << ":\n";                      // Заголовок для верхнего левого угла
    for (int i = 0; i < K; ++i) {                                            // Печать K строк
        for (int j = 0; j < K; ++j) {                                        // Печать K столбцов
            int v = cell(i, j);                                              // Значение в углу
            if (v >= INF / 2) std::cout << std::setw(6) << "INF";            // INF если недостижимо
            else              std::cout << std::setw(6) << v;                // Иначе расстояние
        }                                                                    // Конец цикла по j
        std::cout << "\n";                                                   // Новая строка после i-й строки угла
    }                                                                        // Конец цикла по i

    // Print bottom-right corner                                              // Комментарий: печать нижнего правого угла
    std::cout << "Bottom-right " << K << "x" << K << ":\n";                  // Заголовок для нижнего правого угла
    for (int i = N - K; i < N; ++i) {                                        // Последние K строк
        for (int j = N - K; j < N; ++j) {                                    // Последние K столбцов
            int v = cell(i, j);                                              // Значение в нижнем правом углу
            if (v >= INF / 2) std::cout << std::setw(6) << "INF";            // Если INF — печатаем INF
            else              std::cout << std::setw(6) << v;                // Иначе печатаем расстояние
        }                                                                    // Конец цикла по j
        std::cout << "\n";                                                   // Новая строка после строки угла
    }                                                                        // Конец цикла по i

    // Optional checksum (helps verify results without printing everything)    // Дополнительно: checksum, чтобы проверять корректность без полного вывода
    std::uint64_t checksum = 0;                                              // Переменная для контрольной суммы
    for (int i = 0; i < N; ++i) {                                            // Перебор всех строк для checksum
        for (int j = 0; j < N; ++j) {                                        // Перебор всех столбцов для checksum
            int v = cell(i, j);                                              // Берём dist[i][j]
            // map INF to a stable constant so checksum is deterministic       // Комментарий: INF заменяем на фикс. константу для детерминизма
            std::uint64_t x = (v >= INF / 2) ? 0x9e3779b97f4a7c15ULL : (std::uint64_t)(unsigned)v; // Преобразуем значение в uint64
            checksum ^= (x + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2)); // Примитивный hash-mix (похоже на boost hash)
        }                                                                    // Конец цикла по j
    }                                                                        // Конец цикла по i
    std::cout << "Checksum: " << checksum << "\n";                           // Печать checksum
}                                                                            // Конец print_matrix_short

int main(int argc, char** argv) {                                            // Точка входа программы
    MPI_Init(&argc, &argv);                                                  // Инициализация MPI

    int rank = 0, size = 1;                                                  // Инициализация rank/size значениями по умолчанию
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                    // Получаем rank текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);                                    // Получаем общее число процессов

    // -------------------- Parse N --------------------                      // Раздел: чтение N из аргументов
    int N = 8;                                                               // Значение N по умолчанию
    if (argc >= 2) {                                                         // Если пользователь передал N
        N = std::atoi(argv[1]);                                              // Преобразуем argv[1] в int
        if (N < 1) N = 1;                                                    // Защита: N минимум 1
    }                                                                        // Конец обработки аргументов

    // We'll use large number as INF (avoid overflow in addition)             // Комментарий: INF большой, чтобы не было переполнения при сложении
    const int INF = 1'000'000'000;                                           // Значение INF (1e9)

    // -------------------- Build row distribution --------------------       // Раздел: распределение строк по процессам
    std::vector<int> row_counts(size, 0);                                    // Сколько строк получает каждый процесс
    std::vector<int> row_displs(size, 0);                                    // Смещение (первая строка) для каждого процесса

    int base = N / size;                                                     // Базовое число строк на процесс
    int rem  = N % size;                                                     // Остаток строк для распределения

    for (int r = 0; r < size; ++r) {                                         // Проходим по всем процессам
        row_counts[r] = base + (r < rem ? 1 : 0);                             // Первые rem процессов получают +1 строку
    }                                                                        // Конец цикла по row_counts
    row_displs[0] = 0;                                                       // Первый блок начинается с 0
    for (int r = 1; r < size; ++r) {                                         // Смещения для остальных процессов
        row_displs[r] = row_displs[r - 1] + row_counts[r - 1];               // Суммируем длины предыдущих блоков
    }                                                                        // Конец цикла по row_displs

    int local_rows  = row_counts[rank];                                      // Число строк на текущем процессе
    int local_elems = local_rows * N;                                        // Число элементов в локальном блоке (строки * N)

    std::vector<int> dist_local(local_elems, INF);                           // Локальный блок матрицы расстояний, заполненный INF
    std::vector<int> dist_full((size_t)N * (size_t)N, INF);                  // Полная матрица расстояний (будет храниться у всех после Allgatherv)

    // -------------------- Rank 0 generates adjacency matrix -------------------- // Раздел: генерация графа на rank 0
    std::vector<int> G;                                                      // Глобальная матрица смежности (хранится только на root перед Scatterv)
    double gen_time = 0.0;                                                   // Время генерации графа
    if (rank == 0) {                                                         // Только процесс 0 генерирует матрицу
        double t0 = MPI_Wtime();                                             // Старт таймера генерации

        G.resize((size_t)N * (size_t)N, INF);                                // Выделяем матрицу N×N и заполняем INF
        std::mt19937 rng(12345);                                             // Генератор случайных чисел (фикс seed для повторяемости)
        std::uniform_int_distribution<int> wdist(1, 20);                     // Распределение весов ребра [1..20]
        std::uniform_real_distribution<double> p(0.0, 1.0);                  // Распределение вероятности [0..1)

        double edge_prob = 0.35;                                             // Вероятность существования ребра

        for (int i = 0; i < N; ++i) {                                        // Цикл по вершинам i
            for (int j = 0; j < N; ++j) {                                    // Цикл по вершинам j
                if (i == j) {                                                // Если диагональ
                    G[(size_t)i * N + j] = 0;                                // Расстояние от вершины к себе = 0
                } else {                                                     // Если разные вершины
                    G[(size_t)i * N + j] = (p(rng) < edge_prob) ? wdist(rng) : INF; // С вероятностью edge_prob ставим вес, иначе INF
                }                                                            // Конец if/else
            }                                                                // Конец цикла по j
        }                                                                    // Конец цикла по i

        double t1 = MPI_Wtime();                                             // Конец таймера генерации
        gen_time = t1 - t0;                                                  // Сохраняем длительность генерации
    }                                                                        // Конец if rank==0

    // -------------------- Prepare Scatterv meta --------------------         // Раздел: подготовка counts/displs для Scatterv/Allgatherv
    std::vector<int> counts_elems(size, 0);                                  // Кол-во элементов (int) для каждого процесса
    std::vector<int> displs_elems(size, 0);                                  // Смещения по элементам (int) для каждого процесса
    for (int r = 0; r < size; ++r) {                                         // Цикл по процессам
        counts_elems[r] = row_counts[r] * N;                                 // Элементы = строки * N
        displs_elems[r] = row_displs[r] * N;                                 // Смещение = смещение строк * N
    }                                                                        // Конец цикла

    // -------------------- TIMING variables --------------------              // Раздел: переменные тайминга
    double t0_total = 0.0, t1_total = 0.0;                                   // Таймер общего времени (начало/конец)
    double t_sc0 = 0.0, t_sc1 = 0.0;      // Scatterv                          // Таймер Scatterv (start/end)
    double t_ag0 = 0.0, t_ag1 = 0.0;      // init Allgatherv                   // Таймер первичного Allgatherv (start/end)
    double t_loop0 = 0.0, t_loop1 = 0.0;  // main loop (bcast+update+allgather) // Таймер основного цикла (start/end)
    double t_pr0 = 0.0, t_pr1 = 0.0;      // print (rank0 only, still measure) // Таймер печати (меряем, хотя выводит только root)

    MPI_Barrier(MPI_COMM_WORLD);                                             // Синхронизация перед общим таймером
    t0_total = MPI_Wtime();                                                  // Старт общего таймера

    // -------------------- Scatterv timing --------------------               // Раздел: тайминг Scatterv
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед Scatterv для честного измерения
    t_sc0 = MPI_Wtime();                                                     // Start Scatterv timer

    MPI_Scatterv(                                                            // Scatterv: раздать блоки строк матрицы G
        rank == 0 ? G.data() : nullptr,                                      // Отправляющий буфер только у root
        counts_elems.data(),                                                 // Сколько int отправить каждому процессу
        displs_elems.data(),                                                 // Смещения, откуда брать блоки
        MPI_INT,                                                             // Тип данных: int
        dist_local.data(),                                                   // Буфер, куда принимать локальный блок
        local_elems,                                                         // Сколько int принимает текущий процесс
        MPI_INT,                                                             // Тип данных: int
        0,                                                                   // Root процесса
        MPI_COMM_WORLD                                                       // Коммуникатор
    );                                                                       // Конец MPI_Scatterv

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер после Scatterv
    t_sc1 = MPI_Wtime();                                                     // End Scatterv timer

    // -------------------- Init full matrix via Allgatherv -------------------- // Раздел: собрать полную матрицу у всех через Allgatherv
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед Allgatherv
    t_ag0 = MPI_Wtime();                                                     // Start init allgather timer

    MPI_Allgatherv(                                                          // Allgatherv: собрать dist_full у всех процессов
        dist_local.data(),                                                   // Отправляем локальный блок
        local_elems,                                                         // Кол-во отправляемых элементов
        MPI_INT,                                                             // Тип данных
        dist_full.data(),                                                    // Принимающий буфер полной матрицы
        counts_elems.data(),                                                 // Сколько элементов приходит от каждого процесса
        displs_elems.data(),                                                 // Смещения, куда класть блоки
        MPI_INT,                                                             // Тип данных
        MPI_COMM_WORLD                                                       // Коммуникатор
    );                                                                       // Конец MPI_Allgatherv

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер после Allgatherv
    t_ag1 = MPI_Wtime();                                                     // End init allgather timer

    // Buffer for pivot row k                                                  // Комментарий: буфер для k-й строки (pivot row)
    std::vector<int> rowk(N, INF);                                           // Массив rowk хранит строку k длиной N

    // -------------------- Main loop timing --------------------              // Раздел: тайминг основного цикла
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед основным циклом
    t_loop0 = MPI_Wtime();                                                   // Start main loop timer

    for (int k = 0; k < N; ++k) {                                            // Основной Floyd–Warshall: промежуточная вершина k
        int owner = owner_of_row(k, row_counts, row_displs);                 // Определяем, у кого хранится строка k (локально)

        if (rank == owner) {                                                 // Если текущий процесс владеет строкой k
            int local_k = k - row_displs[rank];                              // Переводим глобальный индекс k в локальный индекс
            for (int j = 0; j < N; ++j) {                                    // Копируем строку k в rowk
                rowk[j] = dist_local[(size_t)local_k * N + j];               // rowk[j] = dist[k][j] из локального блока
            }                                                                // Конец копирования строки
        }                                                                    // Конец if owner

        MPI_Bcast(rowk.data(), N, MPI_INT, owner, MPI_COMM_WORLD);           // Рассылаем строку k всем процессам

        for (int li = 0; li < local_rows; ++li) {                            // Перебор локальных строк i (внутри блока процесса)
            int gi  = row_displs[rank] + li;                                 // Глобальный индекс строки i
            int dik = dist_full[(size_t)gi * (size_t)N + (size_t)k];         // Берём d(i,k) из полной матрицы (с прошлого allgather)
            if (dik >= INF) continue;                                        // Если d(i,k)=INF, улучшений через k быть не может

            for (int j = 0; j < N; ++j) {                                    // Перебор всех столбцов j
                int dkj = rowk[j];                                           // Берём d(k,j) из broadcast строки
                if (dkj >= INF) continue;                                    // Если d(k,j)=INF, путь i->k->j невозможен

                long long cand = (long long)dik + (long long)dkj;            // Кандидатное расстояние через k (в long long чтобы не переполнить int)
                int& dij = dist_local[(size_t)li * (size_t)N + (size_t)j];   // Ссылка на локальное d(i,j)
                if (cand < dij) dij = (int)cand;                             // Если путь через k короче — обновляем d(i,j)
            }                                                                // Конец цикла по j
        }                                                                    // Конец цикла по li

        // Requirement: everyone exchanges updated blocks each iteration       // Требование задания: обменивать обновлённые блоки на каждой итерации
        MPI_Allgatherv(                                                      // Allgatherv после каждой k-итерации
            dist_local.data(),                                               // Отправляем обновлённый локальный блок
            local_elems,                                                     // Кол-во отправляемых элементов
            MPI_INT,                                                         // Тип данных
            dist_full.data(),                                                // Собираем обновлённую полную матрицу
            counts_elems.data(),                                             // Кол-во элементов от каждого процесса
            displs_elems.data(),                                             // Смещения
            MPI_INT,                                                         // Тип данных
            MPI_COMM_WORLD                                                   // Коммуникатор
        );                                                                   // Конец MPI_Allgatherv в цикле
    }                                                                        // Конец основного цикла по k

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер после основного цикла
    t_loop1 = MPI_Wtime();                                                   // End main loop timer

    // -------------------- Print timing (keep output short) -------------------- // Раздел: тайминг вывода (и короткий вывод)
    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед печатью
    t_pr0 = MPI_Wtime();                                                     // Start print timer

    print_matrix_short(dist_full, N, INF, rank, size);                       // Печатаем сокращённый результат (только rank 0)

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер после печати (чтобы замер был согласован)
    t_pr1 = MPI_Wtime();                                                     // End print timer

    MPI_Barrier(MPI_COMM_WORLD);                                             // Барьер перед завершением общего таймера
    t1_total = MPI_Wtime();                                                  // End total timer

    // -------------------- Reduce MAX timings across ranks -------------------- // Раздел: сводим времена (берём максимум по процессам)
    double scatter_t = t_sc1 - t_sc0;                                        // Scatterv длительность на текущем rank
    double init_ag_t = t_ag1 - t_ag0;                                        // Init Allgatherv длительность на текущем rank
    double loop_t    = t_loop1 - t_loop0;                                    // Main loop длительность на текущем rank
    double print_t   = t_pr1 - t_pr0;                                        // Print длительность на текущем rank
    double total_t   = t1_total - t0_total;                                  // Total длительность на текущем rank

    double scatter_max = 0, init_ag_max = 0, loop_max = 0, print_max = 0, total_max = 0; // Переменные для max значений на root

    MPI_Reduce(&scatter_t, &scatter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Максимум Scatterv времени по процессам
    MPI_Reduce(&init_ag_t, &init_ag_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Максимум Init Allgatherv времени по процессам
    MPI_Reduce(&loop_t,    &loop_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Максимум Main loop времени по процессам
    MPI_Reduce(&print_t,   &print_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Максимум Print времени по процессам
    MPI_Reduce(&total_t,   &total_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Максимум Total времени по процессам

    if (rank == 0) {                                                         // Только root печатает итоговые тайминги
        std::cout << std::fixed << std::setprecision(6);                     // Формат вывода: fixed и 6 знаков после запятой

        // Required by statement:                                              // Комментарий: требуемый вывод по условию задания
        std::cout << "Execution time: " << total_max << " seconds.\n";       // Печатаем строку "Execution time: ... seconds."

        std::cout << "Timing (max over ranks):\n";                           // Заголовок таблицы таймингов
        std::cout << "  Gen (rank0 only) : " << gen_time     * 1000.0 << " ms\n"; // Время генерации (только root)
        std::cout << "  Scatterv         : " << scatter_max  * 1000.0 << " ms\n"; // Scatterv максимум
        std::cout << "  Init Allgatherv  : " << init_ag_max  * 1000.0 << " ms\n"; // Начальный Allgatherv максимум
        std::cout << "  Main loop        : " << loop_max     * 1000.0 << " ms\n"; // Основной цикл максимум
        std::cout << "  Print            : " << print_max    * 1000.0 << " ms\n"; // Печать максимум
        std::cout << "  Total            : " << total_max    * 1000.0 << " ms\n"; // Общее время максимум
    }                                                                        // Конец if rank==0

    MPI_Finalize();                                                          // Завершение MPI окружения
    return 0;                                                                // Успешное завершение программы
}                                                                            // Конец main
