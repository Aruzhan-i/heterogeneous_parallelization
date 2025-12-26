#include <iostream>   // cout, cin, endl
#include <random>     // random_device, mt19937, uniform_int_distribution
#include <chrono>     // измерение времени
#include <omp.h>      // OpenMP

using namespace std;
using namespace chrono;

// Заполнение массива случайными числами [1..100]
void fill_random(int* a, int n) {
    random_device rd;                           // Источник энтропии
    mt19937 gen(rd());                          // Генератор Mersenne Twister
    uniform_int_distribution<int> dist(1, 100); // Диапазон случайных чисел

    for (int i = 0; i < n; i++) {
        a[i] = dist(gen);                       // Заполняем массив
    }
}

// Последовательное вычисление среднего значения
double mean_seq(const int* a, int n) {
    long long sum = 0;                          // Сумма элементов массива
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return static_cast<double>(sum) / n;        // Возвращаем среднее
}

// Параллельное вычисление среднего значения (OpenMP reduction)
double mean_omp(const int* a, int n) {
    long long sum = 0;                          // Общая сумма

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i];                            // Частичные суммы потоков
    }

    return static_cast<double>(sum) / n;        // Возвращаем среднее
}

int main() {

    // Переключаем кодировку консоли Windows на UTF-8 для корректного отображения русского текста
    system("chcp 65001 > nul");

    int N;                                      // Размер массива

    // Ввод размера массива
    cout << "Введите размер массива: ";
    cin >> N;

    // Проверка корректности ввода
    if (N <= 0) {
        cout << "Ошибка: размер массива должен быть положительным числом." << endl;
        return 1;
    }

    // Выделение динамической памяти
    cout << "\nВыделяется память под массив..." << endl;
    int* a = new int[N];

    // Заполнение массива
    cout << "Массив заполняется случайными числами..." << endl;
    fill_random(a, N);

    // Вывод количества доступных потоков OpenMP
    cout << "\nМаксимальное количество потоков OpenMP: "
         << omp_get_max_threads() << "\n\n";

    // ===============================
    // Последовательное вычисление
    // ===============================
    cout << "Запуск последовательного вычисления среднего значения..." << endl;

    auto t1 = high_resolution_clock::now();
    double avg_seq = mean_seq(a, N);
    auto t2 = high_resolution_clock::now();
    auto time_seq = duration_cast<microseconds>(t2 - t1).count();

    cout << "Последовательное среднее значение = " << avg_seq << endl;
    cout << "Время выполнения последовательного алгоритма = "
         << time_seq << " мкс" << endl;

    // ===============================
    // Параллельное вычисление OpenMP
    // ===============================
    cout << "\nЗапуск параллельного вычисления среднего значения (OpenMP)..." << endl;

    t1 = high_resolution_clock::now();
    double avg_omp = mean_omp(a, N);
    t2 = high_resolution_clock::now();
    auto time_omp = duration_cast<microseconds>(t2 - t1).count();

    cout << "Параллельное среднее значение = " << avg_omp << endl;
    cout << "Время выполнения параллельного алгоритма = "
         << time_omp << " мкс" << endl;

    // Проверка корректности результатов
    if (avg_seq != avg_omp) {
        cout << "\nВНИМАНИЕ: результаты последовательного и параллельного алгоритмов не совпадают!"
             << endl;
    }

    // Освобождение памяти
    cout << "\nОсвобождаем динамически выделенную память." << endl;
    delete[] a;

    // Сообщение о завершении программы
    cout << "Программа завершена корректно." << endl;
    return 0;
}
