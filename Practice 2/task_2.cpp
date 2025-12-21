#include <iostream>
#include <vector>
#include <random>

// Подключение библиотеки chrono.
// Используется для измерения времени выполнения алгоритмов.
#include <chrono>

// Подключение библиотеки OpenMP.
// Необходима для директив #pragma omp parallel for.
#include <omp.h>

// Используем стандартное пространство имён.
using namespace std;

// Используем пространство имён chrono для работы с таймерами.
using namespace chrono;


// Функция генерации массива случайных чисел.
// n — размер массива.
vector<int> generateArray(int n) {

    // Создаём вектор размером n.
    vector<int> a(n);

    // Генератор псевдослучайных чисел Mersenne Twister.
    // Фиксированное зерно (seed) используется для повторяемости экспериментов.
    mt19937 rng(42);

    // Равномерное распределение целых чисел
    // в диапазоне от 0 до 100000.
    uniform_int_distribution<int> dist(0, 100000);

    // Заполняем массив случайными значениями.
    for (int& x : a) {
        x = dist(rng);
    }

    // Возвращаем заполненный массив.
    return a;
}


// Параллельная пузырьковая сортировка.
// Используется odd-even sort — корректная параллельная модификация пузырька.
void bubbleSortParallel(vector<int>& a) {

    // Получаем размер массива.
    int n = a.size();

    // Внешний цикл задаёт фазы сортировки.
    // Количество фаз равно размеру массива.
    for (int phase = 0; phase < n; phase++) {

        // Определяем начальный индекс:
        // 0 — чётная фаза, 1 — нечётная.
        int start = phase % 2;

        // Параллельный цикл OpenMP.
        // Каждая итерация обрабатывает независимую пару элементов.
        #pragma omp parallel for
        for (int i = start; i < n - 1; i += 2) {

            // Сравниваем элементы пары и при необходимости меняем местами.
            if (a[i] > a[i + 1]) {
                swap(a[i], a[i + 1]);
            }
        }
        // Неявный барьер OpenMP гарантирует,
        // что все потоки завершили фазу перед следующей.
    }
}


// Параллельная сортировка выбором.
// Внешний цикл остаётся последовательным,
// поиск минимума выполняется параллельно.
void selectionSortParallel(vector<int>& a) {

    // Получаем размер массива.
    int n = a.size();

    // Внешний цикл по позициям массива.
    for (int i = 0; i < n - 1; i++) {

        // Предполагаем, что минимальный элемент находится на позиции i.
        int minIdx = i;

        // Параллельный поиск минимального элемента
        // в неотсортированной части массива.
        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {

            // Если найден элемент меньше текущего минимума.
            if (a[j] < a[minIdx]) {

                // Критическая секция.
                // Защищает изменение minIdx от гонки данных.
                #pragma omp critical
                minIdx = j;
            }
        }

        // Помещаем найденный минимальный элемент на позицию i.
        swap(a[i], a[minIdx]);
    }
}


// Сортировка вставками.
// Алгоритм по природе последовательный,
// поэтому корректная параллелизация невозможна.
void insertionSortParallel(vector<int>& a) {

    // Получаем размер массива.
    int n = a.size();

    // Алгоритм начинается со второго элемента.
    for (int i = 1; i < n; i++) {

        // Элемент, который необходимо вставить.
        int key = a[i];

        // Индекс для движения влево.
        int j = i - 1;

        // Сдвигаем элементы вправо.
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }

        // Вставляем элемент на корректную позицию.
        a[j + 1] = key;
    }
}


int main() {

    // Набор размеров массивов для тестирования.
    vector<int> sizes = {1000, 10000, 100000};

    // Вывод количества потоков OpenMP.
    cout << "OpenMP threads: " << omp_get_max_threads() << endl;

    // Цикл по каждому размеру массива.
    for (int n : sizes) {

        // Генерируем исходный массив.
        vector<int> base = generateArray(n);

        // Создаём копии массива для каждой сортировки.
        vector<int> aBubble = base;
        vector<int> aSelect = base;
        vector<int> aInsert = base;

        // ================= Bubble Sort =================
        auto t1 = high_resolution_clock::now();
        bubbleSortParallel(aBubble);
        auto t2 = high_resolution_clock::now();

        double bubbleTime =
            duration<double, milli>(t2 - t1).count();

        // ================= Selection Sort =================
        auto t3 = high_resolution_clock::now();
        selectionSortParallel(aSelect);
        auto t4 = high_resolution_clock::now();

        double selectionTime =
            duration<double, milli>(t4 - t3).count();

        // ================= Insertion Sort =================
        auto t5 = high_resolution_clock::now();
        insertionSortParallel(aInsert);
        auto t6 = high_resolution_clock::now();

        double insertionTime =
            duration<double, milli>(t6 - t5).count();

        // ================= Вывод результатов =================
        cout << "\nArray size: " << n << endl;
        cout << "Bubble sort (parallel):    " << bubbleTime << " ms\n";
        cout << "Selection sort (parallel): " << selectionTime << " ms\n";
        cout << "Insertion sort:            " << insertionTime << " ms\n";
    }

    // Завершение программы.
    return 0;
}

