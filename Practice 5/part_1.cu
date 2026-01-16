// part_1.cu // Название файла
#include <cuda_runtime.h> // Подключение функций среды исполнения CUDA
#include <iostream> // Подключение стандартного ввода-вывода C++
#include <vector> // Подключение контейнера vector
#include <unordered_set> // Подключение контейнера для хранения уникальных значений
#include <iomanip> // Подключение инструментов манипуляции выводом (форматирование)
#include <cstdlib> // Подключение стандартных функций (atoi, exit)


#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call); /* Вызов функции и сохранение кода */ \
  if (err != cudaSuccess) { /* Проверка на ошибку */              \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
    std::exit(1);           /* Выход при ошибке */                \
  }                                                               \
} while(0) // Комментарий здесь допустим, так как это конец макроса


// =====================
// Параллельный стек (LIFO)
// top = текущий размер (0..capacity)
// =====================
struct Stack { // Определение структуры стека
  int* data; // Указатель на массив данных в памяти GPU
  int* top;       // Указатель на индекс вершины в глобальной памяти
  int capacity; // Максимальная вместимость стека

  __device__ void init(int* buffer, int* top_ptr, int size) { // Метод инициализации на GPU
    data = buffer; // Присваивание буфера данных
    top = top_ptr; // Присваивание указателя на вершину
    capacity = size; // Установка емкости
    if (threadIdx.x == 0 && blockIdx.x == 0) { // Только самый первый поток выполняет сброс
      *top = 0; // Установка индекса вершины в ноль (пустой стек)
    } // Конец условия для первого потока
    __threadfence(); // Гарантия видимости записи в top для всех остальных потоков
  } // Конец метода init

  __device__ bool push(int value) { // Метод добавления элемента
    // atomicAdd возвращает старое значение top => idx = старый размер
    int idx = atomicAdd(top, 1); // Атомарное инкрементирование индекса вершины
    if (idx < capacity) { // Проверка на переполнение
      data[idx] = value; // Запись значения в ячейку массива
      return true; // Успешное добавление
    } else { // Если стек полон
      // переполнение: откатываем top назад
      atomicSub(top, 1); // Атомарное уменьшение счетчика обратно
      return false; // Добавление не удалось
    } // Конец условия проверки емкости
  } // Конец метода push

  __device__ bool pop(int* value) { // Метод извлечения элемента
    // atomicSub возвращает старое значение top
    // после декремента новый размер = old-1, индекс верхнего элемента = old-1
    int old = atomicSub(top, 1); // Атомарное уменьшение индекса
    int idx = old - 1; // Вычисление индекса извлекаемого элемента
    if (idx >= 0) { // Если стек не был пустым
      *value = data[idx]; // Чтение значения из массива
      return true; // Успешное извлечение
    } else { // Если индекс стал отрицательным (стек пуст)
      // стек был пуст: откатываем top обратно
      atomicAdd(top, 1); // Возвращаем счетчик в исходное состояние (0)
      return false; // Извлечение не удалось
    } // Конец условия проверки на пустоту
  } // Конец метода pop
}; // Конец структуры Stack

// ---------------------
// Kernel: параллельные push
// ---------------------
__global__ void stack_push_kernel(int* stack_buf, int* top_ptr, int cap,
                                  int n_push,
                                  int* push_ok) // Ядро для добавления элементов
{ // Начало функции ядра
  Stack st; // Локальный объект структуры стека для потока
  st.init(stack_buf, top_ptr, cap); // Инициализация (сброс только первым потоком)

  int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального ID потока
  if (tid < n_push) { // Проверка границ (не выходим ли за число элементов)
    // кладём в стек tid (уникальные значения)
    push_ok[tid] = st.push(tid) ? 1 : 0; // Сохранение статуса операции в массив результатов
  } // Конец условия по ID потока
} // Конец функции ядра push

// ---------------------
// Kernel: параллельные pop
// ---------------------
__global__ void stack_pop_kernel(int* stack_buf, int* top_ptr, int cap,
                                 int n_pop,
                                 int* pop_out,
                                 int* pop_ok) // Ядро для извлечения элементов
{ // Начало функции ядра
  Stack st; // Локальный объект структуры стека
  // ВНИМАНИЕ: init тут НЕ сбрасывает top, иначе потеряем данные.
  st.data = stack_buf; // Настройка указателя на данные
  st.top = top_ptr; // Настройка указателя на вершину
  st.capacity = cap; // Настройка емкости

  int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление ID потока
  if (tid < n_pop) { // Проверка границ
    int v = -1; // Переменная для хранения извлеченного значения
    bool ok = st.pop(&v); // Попытка извлечь значение из стека
    pop_out[tid] = v; // Сохранение значения в выходной массив
    pop_ok[tid]  = ok ? 1 : 0; // Сохранение статуса (успех/провал)
  } // Конец условия по ID
} // Конец функции ядра pop

// ---------------------
// Простая функция проверки:
// 1) считаем сколько push успешно
// 2) считаем сколько pop успешно
// 3) проверяем, что popped значения входят в набор pushed (уникальность optional)
// ---------------------
static void verify_stack(const std::vector<int>& push_ok,
                         const std::vector<int>& pop_ok,
                         const std::vector<int>& pop_out,
                         int n_push) // Функция верификации результатов на CPU
{ // Начало функции проверки
  int pushed = 0; // Счетчик успешных добавлений
  for (int x : push_ok) pushed += x; // Суммирование флагов успеха push

  int popped = 0; // Счетчик успешных извлечений
  for (int x : pop_ok) popped += x; // Суммирование флагов успеха pop

  std::unordered_set<int> popped_vals; // Множество для проверки уникальности
  int bad_range = 0; // Счетчик значений, вышедших за ожидаемый диапазон
  for (size_t i = 0; i < pop_out.size(); ++i) { // Цикл по результатам извлечения
    if (pop_ok[i]) { // Если извлечение было успешным
      int v = pop_out[i]; // Получаем значение
      if (v < 0 || v >= n_push) bad_range++; // Проверка на корректность данных
      popped_vals.insert(v); // Добавление в набор для проверки уникальности
    } // Конец условия проверки успеха
  } // Конец цикла по элементам

  std::cout << "Проверка корректности:\n"; // Вывод заголовка отчета
  std::cout << "  Успешных push: " << pushed << "\n"; // Печать кол-во успешных push
  std::cout << "  Успешных pop : " << popped << "\n"; // Печать кол-во успешных pop
  std::cout << "  Уникальных извлечённых значений: " << popped_vals.size() << "\n"; // Печать размера множества
  std::cout << "  Значений вне диапазона [0.." << (n_push-1) << "]: " << bad_range << "\n"; // Печать ошибок диапазона
} // Конец функции верификации

// ---------------------
// Замер времени CUDA kernel (через events)
// ---------------------
static float time_kernel(void (*launcher)(), int iters = 50) { // Функция профилирования
  cudaEvent_t start, stop; // События CUDA для отсчета времени
  CUDA_CHECK(cudaEventCreate(&start)); // Создание события начала
  CUDA_CHECK(cudaEventCreate(&stop)); // Создание события конца

  // прогрев
  launcher(); // Запуск функции-лаунчера для инициализации GPU
  CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения всех операций

  CUDA_CHECK(cudaEventRecord(start)); // Фиксация времени начала
  for (int i = 0; i < iters; ++i) launcher(); // Цикл запусков для усреднения
  CUDA_CHECK(cudaEventRecord(stop)); // Фиксация времени окончания

  CUDA_CHECK(cudaEventSynchronize(stop)); // Ожидание завершения последнего запуска
  float ms = 0.0f; // Переменная для результата в миллисекундах
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Расчет разницы во времени

  CUDA_CHECK(cudaEventDestroy(start)); // Удаление события начала
  CUDA_CHECK(cudaEventDestroy(stop)); // Удаление события конца
  return ms / iters; // Возврат среднего времени выполнения
} // Конец функции профилирования

int main(int argc, char** argv) { // Точка входа в программу
  // Параметры (можно менять)
  int n_push = 1 << 20;   // Количество потоков-производителей (1048576)
  int cap    = 1 << 18;   // Максимальная емкость стека (262144)
  int n_pop  = cap;       // Количество потоков-потребителей

  if (argc >= 2) n_push = std::atoi(argv[1]); // Считывание n_push из аргументов командной строки
  if (argc >= 3) cap    = std::atoi(argv[2]); // Считывание cap из аргументов командной строки
  if (argc >= 4) n_pop  = std::atoi(argv[3]); // Считывание n_pop из аргументов командной строки

  std::cout << "=== Практика 5: Параллельный стек (CUDA) ===\n"; // Информационный вывод
  std::cout << "n_push=" << n_push << ", cap=" << cap << ", n_pop=" << n_pop << "\n"; // Вывод параметров

  // Device memory
  int* d_stack = nullptr; // Указатель на буфер стека на GPU
  int* d_top   = nullptr; // Указатель на индекс вершины на GPU

  int* d_push_ok = nullptr; // Указатель на массив статусов push на GPU
  int* d_pop_ok  = nullptr; // Указатель на массив статусов pop на GPU
  int* d_pop_out = nullptr; // Указатель на массив извлеченных данных на GPU

  CUDA_CHECK(cudaMalloc(&d_stack,   cap * sizeof(int))); // Выделение памяти под стек
  CUDA_CHECK(cudaMalloc(&d_top,     sizeof(int))); // Выделение памяти под счетчик вершины
  CUDA_CHECK(cudaMalloc(&d_push_ok, n_push * sizeof(int))); // Выделение памяти под статусы push
  CUDA_CHECK(cudaMalloc(&d_pop_ok,  n_pop  * sizeof(int))); // Выделение памяти под статусы pop
  CUDA_CHECK(cudaMalloc(&d_pop_out, n_pop  * sizeof(int))); // Выделение памяти под результаты pop

  int threads = 256; // Количество потоков в одном блоке
  int blocks_push = (n_push + threads - 1) / threads; // Расчет количества блоков для push
  int blocks_pop  = (n_pop  + threads - 1) / threads; // Расчет количества блоков для pop

  // Лаунчеры для замера
  auto launch_push = [&](){ // Лямбда-функция для запуска ядра push
    stack_push_kernel<<<blocks_push, threads>>>(d_stack, d_top, cap, n_push, d_push_ok); // Вызов ядра
  }; // Конец лямбды push
  auto launch_pop = [&](){ // Лямбда-функция для запуска ядра pop
    stack_pop_kernel<<<blocks_pop, threads>>>(d_stack, d_top, cap, n_pop, d_pop_out, d_pop_ok); // Вызов ядра
  }; // Конец лямбды pop

  // 1) Выполняем push один раз
  launch_push(); // Заполнение стека
  CUDA_CHECK(cudaDeviceSynchronize()); // Синхронизация GPU и CPU

  // 2) Выполняем pop один раз
  launch_pop(); // Извлечение данных из стека
  CUDA_CHECK(cudaDeviceSynchronize()); // Синхронизация GPU и CPU

  // Копируем результаты для проверки
  std::vector<int> h_push_ok(n_push); // Буфер на хосте для статусов push
  std::vector<int> h_pop_ok(n_pop); // Буфер на хосте для статусов pop
  std::vector<int> h_pop_out(n_pop); // Буфер на хосте для данных pop

  CUDA_CHECK(cudaMemcpy(h_push_ok.data(), d_push_ok, n_push * sizeof(int), cudaMemcpyDeviceToHost)); // Копирование push_ok с GPU
  CUDA_CHECK(cudaMemcpy(h_pop_ok.data(),  d_pop_ok,  n_pop  * sizeof(int), cudaMemcpyDeviceToHost)); // Копирование pop_ok с GPU
  CUDA_CHECK(cudaMemcpy(h_pop_out.data(), d_pop_out, n_pop  * sizeof(int), cudaMemcpyDeviceToHost)); // Копирование результатов с GPU

  verify_stack(h_push_ok, h_pop_ok, h_pop_out, n_push); // Вызов верификации

  // 3) Замер производительности
  // Для честного замера каждый раз заново делаем push (он сбрасывает top в init)
  float push_ms = time_kernel(+[](){}, 1); // Инициализация переменной-заглушки

  // Трюк: оборачиваем в статик-указатель нельзя, поэтому просто отдельно считаем:
  auto push_launcher = [&](){ // Повторное определение лямбды для push (из-за области видимости)
    stack_push_kernel<<<blocks_push, threads>>>(d_stack, d_top, cap, n_push, d_push_ok); // Вызов ядра
  }; // Конец лямбды
  auto pop_launcher = [&](){ // Повторное определение лямбды для pop
    stack_pop_kernel<<<blocks_pop, threads>>>(d_stack, d_top, cap, n_pop, d_pop_out, d_pop_ok); // Вызов ядра
  }; // Конец лямбды

  // Замер push
  { // Начало блока замера push
    cudaEvent_t start, stop; // Локальные события
    CUDA_CHECK(cudaEventCreate(&start)); // Создание
    CUDA_CHECK(cudaEventCreate(&stop)); // Создание

    // прогрев
    push_launcher(); // Разогревочный запуск
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание

    int iters = 50; // Количество итераций
    CUDA_CHECK(cudaEventRecord(start)); // Старт таймера
    for(int i=0;i<iters;i++) push_launcher(); // Цикл запусков ядра push
    CUDA_CHECK(cudaEventRecord(stop)); // Стоп таймера
    CUDA_CHECK(cudaEventSynchronize(stop)); // Синхронизация по стоп-событию

    float ms = 0.0f; // Переменная времени
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Вычисление времени
    push_ms = ms / iters; // Среднее время выполнения push

    CUDA_CHECK(cudaEventDestroy(start)); // Очистка события
    CUDA_CHECK(cudaEventDestroy(stop)); // Очистка события
  } // Конец блока замера push

  // Перед замером pop нужно, чтобы стек был наполнен
  push_launcher(); // Наполнение стека перед тестом pop
  CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание

  float pop_ms = 0.0f; // Переменная для времени pop
  { // Начало блока замера push+pop
    cudaEvent_t start, stop; // Локальные события
    CUDA_CHECK(cudaEventCreate(&start)); // Создание
    CUDA_CHECK(cudaEventCreate(&stop)); // Создание

    int iters = 50; // Количество итераций

    // прогрев: делаем push+pop, потому что pop опустошает стек
    push_launcher(); CUDA_CHECK(cudaDeviceSynchronize()); // Наполнение для прогрева
    pop_launcher();  CUDA_CHECK(cudaDeviceSynchronize()); // Извлечение для прогрева

    CUDA_CHECK(cudaEventRecord(start)); // Старт таймера
    for(int i=0;i<iters;i++){ // Цикл замера
      push_launcher(); // Обязательный push, иначе pop будет работать с пустым стеком
      pop_launcher(); // Собственно pop
    } // Конец цикла
    CUDA_CHECK(cudaEventRecord(stop)); // Стоп таймера
    CUDA_CHECK(cudaEventSynchronize(stop)); // Синхронизация

    float ms = 0.0f; // Переменная времени
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Вычисление общего времени
    pop_ms = ms / iters; // Среднее время за итерацию (push+pop)

    CUDA_CHECK(cudaEventDestroy(start)); // Очистка
    CUDA_CHECK(cudaEventDestroy(stop)); // Очистка
  } // Конец блока замера

  std::cout << "\nЗамеры (среднее на итерацию):\n"; // Вывод заголовка
  std::cout << "  push kernel: " << std::fixed << std::setprecision(4) << push_ms << " мс\n"; // Результат для push
  std::cout << "  push+pop   : " << std::fixed << std::setprecision(4) << pop_ms  << " мс\n"; // Результат для push+pop

  // cleanup
  CUDA_CHECK(cudaFree(d_stack)); // Освобождение памяти стека на GPU
  CUDA_CHECK(cudaFree(d_top)); // Освобождение памяти вершины на GPU
  CUDA_CHECK(cudaFree(d_push_ok)); // Освобождение памяти push_ok
  CUDA_CHECK(cudaFree(d_pop_ok)); // Освобождение памяти pop_ok
  CUDA_CHECK(cudaFree(d_pop_out)); // Освобождение памяти pop_out

  return 0; // Возврат успешного завершения программы
} // Конец функции main