// OpenMP: sum, mean, variance + timing with omp_get_wtime()              // Описание: программа считает сумму, среднее, дисперсию и время выполнения с OpenMP

#include <omp.h>                                                         // Подключаем OpenMP (omp_get_wtime, omp_set_num_threads, директивы #pragma omp)
#include <iostream>                                                      // Подключаем ввод/вывод (std::cout)
#include <vector>                                                        // Подключаем контейнер std::vector для хранения массива
#include <random>                                                        // Подключаем генераторы случайных чисел (mt19937_64, uniform_real_distribution)
#include <cmath>                                                         // Подключаем математику (например, для вычислений; здесь напрямую почти не используется)
#include <iomanip>                                                       // Подключаем форматирование вывода (fixed, setprecision)

int main(int argc, char** argv) {                                        // Точка входа: argc — кол-во аргументов, argv — массив строк аргументов
    const long long N = (argc > 1) ? std::atoll(argv[1]) : 50'000'000;   // Размер массива: берём из argv[1], иначе 50M по умолчанию
    int threads = (argc > 2) ? std::atoi(argv[2]) : omp_get_max_threads();// Кол-во потоков: argv[2], иначе максимум потоков на машине

    omp_set_num_threads(threads);                                        // Устанавливаем число потоков OpenMP для параллельных регионов

    std::vector<double> a(N);                                            // Выделяем массив a длины N (double)

    // ---------- Init data (sequential) ----------                        // Заголовок блока: инициализация данных в одном потоке (последовательно)
    double t0 = omp_get_wtime();                                         // Засекаем старт времени инициализации (wall time)
    std::mt19937_64 rng(42);                                             // Создаём генератор случайных чисел (64-bit Mersenne Twister) с seed=42
    std::uniform_real_distribution<double> dist(0.0, 1.0);               // Равномерное распределение на [0, 1]
    for (long long i = 0; i < N; ++i) a[i] = dist(rng);                  // Заполняем массив псевдослучайными числами последовательно
    double t1 = omp_get_wtime();                                         // Засекаем конец времени инициализации

    // ---------- Parallel: sum ----------                                 // Заголовок блока: параллельное вычисление суммы элементов массива
    double sum = 0.0;                                                   // Переменная для суммы (будет редуцироваться между потоками)
    double t2 = omp_get_wtime();                                        // Засекаем старт параллельной секции суммирования
    #pragma omp parallel for reduction(+:sum)                            // Параллельный for: каждая нить суммирует часть, затем reduction складывает в sum
    for (long long i = 0; i < N; ++i) sum += a[i];                       // Суммируем элементы массива (работа делится по потокам)
    double t3 = omp_get_wtime();                                        // Засекаем конец параллельной секции суммирования

    double mean = sum / (double)N;                                      // Среднее значение: sum / N

    // ---------- Parallel: variance (population variance) ----------       // Заголовок блока: параллельное вычисление дисперсии (генеральной, делим на N)
    double var_sum = 0.0;                                               // Сумма квадратов отклонений (для reduction)
    double t4 = omp_get_wtime();                                        // Засекаем старт параллельной секции дисперсии
    #pragma omp parallel for reduction(+:var_sum)                        // Параллельный for с редукцией: суммируем вклад каждого потока в var_sum
    for (long long i = 0; i < N; ++i) {                                 // Цикл по всем элементам массива (параллельно)
        double d = a[i] - mean;                                         // Отклонение элемента от среднего
        var_sum += d * d;                                               // Добавляем квадрат отклонения к общей сумме var_sum
    }                                                                   // Конец тела цикла for
    double t5 = omp_get_wtime();                                        // Засекаем конец параллельной секции дисперсии

    double variance = var_sum / (double)N;                              // Дисперсия (population variance): var_sum / N

    // ---------- Report ----------                                         // Заголовок блока: вывод результатов и времени
    double t_init = t1 - t0;                                            // Время инициализации (последовательная часть)
    double t_sum  = t3 - t2;                                            // Время параллельного суммирования
    double t_var  = t5 - t4;                                            // Время параллельного вычисления дисперсии
    double t_total = (t5 - t0);                                         // Общее время: от начала init до конца variance

    std::cout << std::fixed << std::setprecision(6);                    // Формат вывода: фиксированная точка + 6 знаков после запятой
    std::cout << "N=" << N << "  threads=" << threads << "\n";          // Печатаем размер массива и число потоков
    std::cout << "sum=" << sum << "  mean=" << mean << "  variance=" << variance << "\n\n"; // Печатаем результаты вычислений
    std::cout << "Timing (s):\n";                                       // Заголовок таблицы времени (секунды)
    std::cout << "  init(sequential) : " << t_init << "\n";             // Время инициализации (последовательное)
    std::cout << "  sum(parallel)    : " << t_sum  << "\n";             // Время параллельной суммы
    std::cout << "  var(parallel)    : " << t_var  << "\n";             // Время параллельной дисперсии
    std::cout << "  total            : " << t_total << "\n";            // Полное время выполнения

    // Estimated fractions for Amdahl (based on measured times)            // Заголовок: оценка долей для закона Амдала по замеренным временам
    double parallel_time = t_sum + t_var;                               // Параллельное время = сумма времени 2 параллельных регионов
    double serial_time   = t_total - parallel_time;                     // Последовательное время = всё остальное (init + оверхед)
    double P = parallel_time / t_total;                                 // Доля параллельной части P
    double S = serial_time / t_total;                                   // Доля последовательной части S

    std::cout << "\nEstimated fractions:\n";                            // Заголовок блока вывода долей
    std::cout << "  Parallel fraction P = " << P << "\n";               // Печатаем оценку P
    std::cout << "  Serial   fraction S = " << S << "\n";               // Печатаем оценку S

    return 0;                                                           // Завершаем программу успешно
}                                                                       // Конец функции main
