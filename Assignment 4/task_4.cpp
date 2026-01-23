// task_4_mpi_scan.cpp                                              // Имя файла: Task 4 (MPI scan)
 // Assignment 4 - Task 4 (25 pts.)                                 // Задание 4, задача 4 (25 баллов)
 // MPI distributed array processing: prefix sum (inclusive scan).   // MPI обработка массива: префиксная сумма (inclusive scan)
 // Steps: Scatterv -> local scan -> Exscan offsets -> add offsets -> Gatherv. // Шаги: Scatterv -> локальный scan -> Exscan offset -> прибавить offset -> Gatherv
 // Measures execution time for P=2,4,8 processes.                   // Измеряет время для P=2,4,8 процессов
 //
 // Build (Linux/macOS):                                             // Сборка (Linux/macOS)
 //   mpicxx -O2 task_4_mpi_scan.cpp -o task_4                        // Команда компиляции
 //
 // Run examples:                                                    // Примеры запуска
 //   mpirun -np 2 ./task_4                                          // Запуск на 2 процессах
 //   mpirun -np 4 ./task_4                                          // Запуск на 4 процессах
 //   mpirun -np 8 ./task_4                                          // Запуск на 8 процессах
 //
 // Optional: set array size:                                        // Опционально: задать размер массива
 //   mpirun -np 4 ./task_4 1000000                                  // Пример с аргументом N

#include <mpi.h>                                                     // Подключаем MPI API
#include <cstdio>                                                    // std::printf
#include <cstdlib>                                                   // std::atoi, std::rand, std::srand
#include <vector>                                                    // std::vector
#include <numeric>                                                   // (не используется напрямую тут, но подключено)
#include <cmath>                                                     // std::fabs

static void inclusive_scan_local(const std::vector<float>& in, std::vector<float>& out) { // Локальный inclusive scan на одном процессе
    if (in.empty()) return;                                          // Если вход пустой — выходим
    out.resize(in.size());                                           // Выход под размер входа
    out[0] = in[0];                                                  // Первый элемент = первый вход (inclusive)
    for (size_t i = 1; i < in.size(); ++i) out[i] = out[i - 1] + in[i]; // Префиксная сумма по локальному чанку
}                                                                    // Конец inclusive_scan_local

static bool verify(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-3f) { // Проверка массивов на равенство с eps
    if (a.size() != b.size()) return false;                          // Если размеры разные — сразу false
    for (size_t i = 0; i < a.size(); ++i) {                          // Проходим по всем элементам
        if (std::fabs(a[i] - b[i]) > eps) return false;              // Если разница больше eps — ошибка
    }                                                                // Конец цикла
    return true;                                                     // Иначе всё ок
}                                                                    // Конец verify

int main(int argc, char** argv) {                                    // Точка входа, аргументы для MPI_Init
    MPI_Init(&argc, &argv);                                          // Инициализация MPI окружения

    int rank = 0, size = 1;                                          // rank = номер процесса, size = всего процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                            // Узнаём rank текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);                            // Узнаём общее число процессов

    // Default N = 1,000,000 (as in tasks 2-3)                        // Значение N по умолчанию
    int N = 1000000;                                                 // Размер полного массива
    if (argc >= 2) N = std::atoi(argv[1]);                           // Если передали аргумент — читаем N из argv[1]
    if (N < 1) N = 1;                                                // Защита: N минимум 1

    // Root prepares full input                                       // Root (rank 0) создаёт полный входной массив
    std::vector<float> full_in;                                      // Полный вход на root
    if (rank == 0) {                                                 // Только root
        full_in.resize(N);                                           // Выделяем N элементов
        std::srand(42);                                              // Seed для воспроизводимости
        for (int i = 0; i < N; ++i) {                                // Заполняем весь массив
            full_in[i] = float((std::rand() % 10) + 1);              // Значения 1..10
        }                                                            // Конец цикла заполнения
    }                                                                // Конец if rank==0

    // Prepare counts + displacements for Scatterv/Gatherv             // Готовим counts/displs для Scatterv/Gatherv
    std::vector<int> counts(size), displs(size);                     // counts[p]=сколько элементов процессу p, displs[p]=смещение
    int base = N / size;                                             // Базовое количество элементов на процесс
    int rem  = N % size;                                             // Остаток (первые rem процессов получат +1)

    for (int p = 0; p < size; ++p) {                                 // Заполняем counts и displs для всех процессов
        counts[p] = base + (p < rem ? 1 : 0);                        // Распределение: первые rem процессов получают на 1 элемент больше
        displs[p] = (p == 0) ? 0 : (displs[p - 1] + counts[p - 1]);  // Смещение: префиксная сумма counts
    }                                                                // Конец цикла p

    int local_n = counts[rank];                                      // Сколько элементов у текущего процесса
    std::vector<float> local_in(local_n), local_out(local_n);        // Локальный вход и локальный выход (scan)

    // ---------------- Timing starts (parallel section) ---------------- // Старт измерения (параллельная часть)
    MPI_Barrier(MPI_COMM_WORLD);                                     // Барьер: синхронизируем процессы перед таймером
    double t0 = MPI_Wtime();                                         // Время старта по MPI_Wtime()

    // 1) Scatter array chunks                                         // Шаг 1: распределяем чанки массива по процессам
    MPI_Scatterv(rank == 0 ? full_in.data() : nullptr,               // sendbuf: только root отправляет full_in, остальные nullptr
                 counts.data(), displs.data(), MPI_FLOAT,            // counts/displs и тип данных для отправки
                 local_in.data(), local_n, MPI_FLOAT,                // recvbuf: локальный вход, сколько принимаем, тип
                 0, MPI_COMM_WORLD);                                 // root=0, коммуникатор world

    // 2) Local inclusive scan                                         // Шаг 2: локальный scan на каждом процессе
    inclusive_scan_local(local_in, local_out);                       // Выполняем scan на своём чанке

    // 3) Compute local sum (last element of local scan)               // Шаг 3: локальная сумма = последний элемент локального scan
    float local_sum = (local_n > 0) ? local_out.back() : 0.0f;        // Если локально есть элементы — берем последний, иначе 0

    // 4) Compute offset = sum of all previous ranks                   // Шаг 4: offset = сумма всех предыдущих процессов
    float offset = 0.0f;                                             // Переменная offset
    // Exscan gives sum of ranks < current rank (rank0 gets undefined -> set to 0) // Exscan даёт сумму предыдущих рангов (rank0 неопределён)
    MPI_Exscan(&local_sum, &offset, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // Exscan: суммируем local_sum по процессам слева
    if (rank == 0) offset = 0.0f;                                    // Для rank0 offset должен быть 0

    // 5) Add offset to local scan results                             // Шаг 5: прибавляем offset к каждому локальному элементу
    for (int i = 0; i < local_n; ++i) local_out[i] += offset;        // local_out становится частью глобального scan

    // 6) Gather result                                                // Шаг 6: собираем полный результат на root
    std::vector<float> full_out;                                     // Полный выходной массив на root
    if (rank == 0) full_out.resize(N);                               // Только root выделяет память под N

    MPI_Gatherv(local_out.data(), local_n, MPI_FLOAT,                // Отправляем локальный результат
                rank == 0 ? full_out.data() : nullptr,               // Приёмный буфер только у root, иначе nullptr
                counts.data(), displs.data(), MPI_FLOAT,             // counts/displs для сборки и тип
                0, MPI_COMM_WORLD);                                  // root=0, коммуникатор world

    MPI_Barrier(MPI_COMM_WORLD);                                     // Барьер: синхронизация перед окончанием таймера
    double t1 = MPI_Wtime();                                         // Время конца
    double elapsed = t1 - t0;                                        // Локальное время выполнения
    // ---------------- Timing ends ----------------                     // Конец измерения

    // Reduce to get max time (реальное время параллельного выполнения) // Берём максимум по ранкам как реальное время параллельной части
    double max_elapsed = 0.0;                                        // Максимальное время среди процессов
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Reduce max на root

    // Optional correctness check on root (compare with sequential CPU scan) // Проверка корректности на root
    if (rank == 0) {                                                 // Только root делает проверку
        // sequential scan for verification                            // Последовательный scan для сравнения
        std::vector<float> ref(N);                                   // Референсный массив
        ref[0] = full_in[0];                                         // Первый элемент ref
        for (int i = 1; i < N; ++i) ref[i] = ref[i - 1] + full_in[i]; // Последовательный inclusive scan

        bool ok = verify(ref, full_out);                             // Сравниваем ref и MPI результат

        std::printf("============================================================\n"); // Разделитель
        std::printf(" MPI Distributed Scan (inclusive)\n");           // Заголовок
        std::printf(" Array size: %d | Processes: %d\n", N, size);    // Размер и число процессов
        std::printf("------------------------------------------------------------\n"); // Разделитель
        std::printf(" Time (max over ranks): %.6f s\n", max_elapsed); // Печатаем максимальное время (по ранкам)
        std::printf(" Correctness: %s\n", ok ? "OK" : "ERROR");       // Печатаем корректность
        std::printf(" Last element: ref=%.2f  mpi=%.2f\n", ref.back(), full_out.back()); // Последний элемент (общая сумма)
        std::printf("============================================================\n"); // Разделитель
    }                                                                // Конец if rank==0

    MPI_Finalize();                                                  // Завершаем MPI
    return 0;                                                        // Успешный выход
}                                                                    // Конец main
