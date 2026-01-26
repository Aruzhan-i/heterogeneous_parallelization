// task_1_openmp_array.cpp
// Задание 1: Обработка массива на CPU с использованием OpenMP
// Операция: умножение каждого элемента массива на 2
// Размер массива: N = 1 000 000
// Замер времени выполнения

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main() {
    const int N = 1'000'000;
    std::vector<float> A(N);

    // ----------------- Инициализация массива -----------------
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
    }

    // ----------------- Обработка массива с OpenMP -----------------
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        A[i] *= 2.0f;   // обработка элемента массива
    }

    auto end = std::chrono::high_resolution_clock::now();

    // ----------------- Подсчет времени -----------------
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "=== OpenMP array processing ===\n";
    std::cout << "N: " << N << "\n";
    std::cout << "Threads used: " << omp_get_max_threads() << "\n";
    std::cout << "Time (ms): " << elapsed.count() << "\n";

    // Проверка (чтобы убедиться, что реально умножилось)
    std::cout << "Check: A[0] = " << A[0] << ", A[N-1] = " << A[N-1] << "\n";

    return 0;
}

