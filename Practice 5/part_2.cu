// part_2.cu // Название файла
#include <cuda_runtime.h> // Подключение функций среды исполнения CUDA
#include <iostream> // Подключение стандартного потока ввода-вывода
#include <vector> // Подключение контейнера динамический массив
#include <unordered_set> // Подключение хеш-таблицы для хранения уникальных элементов
#include <iomanip> // Подключение средств форматирования вывода
#include <cstdlib> // Подключение стандартной библиотеки (для atoi, exit)

#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call); /* Выполнение функции и получение кода ошибки */ \
  if (err != cudaSuccess) { /* Если произошла ошибка */           \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
    std::exit(1);           /* Выход из программы с ошибкой */    \
  }                                                               \
} while(0) /* Макрос для безопасной проверки выполнения функций CUDA */

// =====================
// Параллельная очередь (FIFO)
// =====================
struct Queue { // Определение структуры параллельной очереди
  int* data; // Указатель на массив данных в памяти устройства
  int* head; // Указатель на индекс начала очереди (голова)
  int* tail; // Указатель на индекс конца очереди (хвост)
  int capacity; // Максимальное количество элементов в очереди

  __device__ void init(int* buffer, int* h, int* t, int cap) { // Метод инициализации на GPU
    data = buffer; // Установка указателя на буфер
    head = h; // Установка указателя на голову
    tail = t; // Установка указателя на хвост
    capacity = cap; // Установка максимальной емкости
    if (threadIdx.x == 0 && blockIdx.x == 0) { // Только самый первый поток выполняет сброс
      *head = 0; // Обнуление индекса головы
      *tail = 0; // Обнуление индекса хвоста
    } // Завершение условия для первого потока
    __threadfence(); // Синхронизация памяти для видимости записи всеми потоками
  } // Конец метода init

  __device__ bool enqueue(int value) { // Метод добавления в очередь
    int pos = atomicAdd(tail, 1); // Атомарное получение позиции и сдвиг хвоста
    if (pos < capacity) { // Проверка, не превышена ли емкость
      data[pos] = value; // Запись значения в свободную ячейку
      return true; // Успешное завершение операции
    } else { // Если очередь переполнена
      atomicSub(tail, 1); // Откат индекса хвоста назад
      return false; // Добавление не удалось
    } // Конец условия проверки емкости
  } // Конец метода enqueue

  __device__ bool dequeue(int* value) { // Метод извлечения из очереди
    int pos = atomicAdd(head, 1); // Атомарное получение позиции и сдвиг головы
    if (pos < *tail) { // Проверка, есть ли данные между head и tail
      *value = data[pos]; // Чтение значения из ячейки
      return true; // Успешное извлечение
    } else { // Если очередь пуста
      atomicSub(head, 1); // Откат индекса головы назад
      return false; // Извлечение не удалось
    } // Конец условия проверки наличия данных
  } // Конец метода dequeue
}; // Конец структуры Queue

// ---------------------
// Kernel enqueue
// ---------------------
__global__ void queue_enqueue_kernel(int* buf, int* head, int* tail, int cap,
                                      int n_ops, int* ok) // Ядро для массового добавления
{ // Начало функции ядра
  Queue q; // Создание локального объекта очереди
  q.init(buf, head, tail, cap); // Инициализация (сброс индексов только потоком 0,0)

  int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального индекса потока
  if (tid < n_ops) // Проверка границ по количеству операций
    ok[tid] = q.enqueue(tid) ? 1 : 0; // Попытка добавления и запись статуса успеха
} // Конец функции ядра enqueue

// ---------------------
// Kernel dequeue
// ---------------------
__global__ void queue_dequeue_kernel(int* buf, int* head, int* tail, int cap,
                                      int n_ops, int* out, int* ok) // Ядро для извлечения
{ // Начало функции ядра
  Queue q; // Создание локального объекта очереди
  q.data = buf; // Настройка указателя на данные
  q.head = head; // Настройка указателя на голову
  q.tail = tail; // Настройка указателя на хвост
  q.capacity = cap; // Установка емкости

  int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального индекса потока
  if (tid < n_ops) { // Проверка границ
    int v = -1; // Переменная для хранения извлеченного значения
    bool res = q.dequeue(&v); // Попытка извлечения
    out[tid] = v; // Сохранение значения (или -1 при неудаче)
    ok[tid]  = res ? 1 : 0; // Запись статуса успеха операции
  } // Конец условия по индексу потока
} // Конец функции ядра dequeue

// ---------------------
// Проверка корректности
// ---------------------
static void verify_queue(const std::vector<int>& enq_ok,
                         const std::vector<int>& deq_ok,
                         const std::vector<int>& deq_out,
                         int n_ops) // Функция проверки результатов на хосте
{ // Начало функции верификации
  int enq = 0; for (int x : enq_ok) enq += x; // Подсчет общего количества успешных enqueue
  int deq = 0; for (int x : deq_ok) deq += x; // Подсчет общего количества успешных dequeue

  std::unordered_set<int> uniq; // Создание множества для контроля уникальности
  int bad_range = 0; // Счетчик значений вне ожидаемого диапазона

  for (size_t i = 0; i < deq_out.size(); ++i) { // Цикл по всем извлеченным данным
    if (deq_ok[i]) { // Если операция извлечения была успешной
      int v = deq_out[i]; // Получаем само значение
      if (v < 0 || v >= n_ops) bad_range++; // Проверка корректности значения
      uniq.insert(v); // Добавление в набор (дубликаты игнорируются)
    } // Конец проверки успеха
  } // Конец цикла по данным

  std::cout << "Проверка корректности очереди:\n"; // Вывод заголовка
  std::cout << "  Успешных enqueue: " << enq << "\n"; // Печать кол-ва добавлений
  std::cout << "  Успешных dequeue: " << deq << "\n"; // Печать кол-ва извлечений
  std::cout << "  Уникальных извлечённых: " << uniq.size() << "\n"; // Проверка на отсутствие потерь/дублей
  std::cout << "  Вне диапазона [0.." << (n_ops-1) << "]: " << bad_range << "\n"; // Печать ошибок данных
} // Конец функции верификации

// ---------------------
// Замер времени enqueue
// ---------------------
static float measure_enqueue(int* buf, int* h, int* t, int cap,
                             int n_ops, int* ok,
                             int blocks, int threads) // Функция профилирования
{ // Начало замера
  cudaEvent_t s, e; // События CUDA для старта и конца замера
  CUDA_CHECK(cudaEventCreate(&s)); // Создание события старта
  CUDA_CHECK(cudaEventCreate(&e)); // Создание события конца

  queue_enqueue_kernel<<<blocks, threads>>>(buf, h, t, cap, n_ops, ok); // Прогревочный запуск
  CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения прогрева

  int iters = 50; // Количество итераций для усреднения
  CUDA_CHECK(cudaEventRecord(s)); // Запись времени начала в поток
  for (int i = 0; i < iters; ++i) // Запуск цикла измерений
    queue_enqueue_kernel<<<blocks, threads>>>(buf, h, t, cap, n_ops, ok); // Вызов ядра
  CUDA_CHECK(cudaEventRecord(e)); // Запись времени окончания
  CUDA_CHECK(cudaEventSynchronize(e)); // Ожидание завершения всех итераций

  float ms = 0.0f; // Переменная для накопления времени
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); // Расчет прошедшего времени
  CUDA_CHECK(cudaEventDestroy(s)); // Удаление события старта
  CUDA_CHECK(cudaEventDestroy(e)); // Удаление события конца

  return ms / iters; // Возврат среднего времени выполнения одной итерации
} // Конец функции замера

int main(int argc, char** argv) { // Точка входа в программу
  int n_ops = 1 << 20; // Количество операций по умолчанию (1048576)
  int cap   = 1 << 18; // Вместимость очереди по умолчанию (262144)

  if (argc >= 2) n_ops = std::atoi(argv[1]); // Считывание n_ops из командной строки
  if (argc >= 3) cap   = std::atoi(argv[2]); // Считывание cap из командной строки

  std::cout << "=== Практика 5. Часть 2: Параллельная очередь (CUDA) ===\n"; // Инфовывод
  std::cout << "n_ops=" << n_ops << ", cap=" << cap << "\n"; // Вывод параметров запуска

  int *d_buf=nullptr, *d_head=nullptr, *d_tail=nullptr; // Указатели для памяти GPU
  int *d_enq_ok=nullptr, *d_deq_ok=nullptr, *d_deq_out=nullptr; // Указатели для результатов на GPU

  CUDA_CHECK(cudaMalloc(&d_buf,    cap * sizeof(int))); // Выделение буфера данных
  CUDA_CHECK(cudaMalloc(&d_head,   sizeof(int))); // Выделение памяти под индекс головы
  CUDA_CHECK(cudaMalloc(&d_tail,   sizeof(int))); // Выделение памяти под индекс хвоста
  CUDA_CHECK(cudaMalloc(&d_enq_ok, n_ops * sizeof(int))); // Память под статус добавления
  CUDA_CHECK(cudaMalloc(&d_deq_ok, n_ops * sizeof(int))); // Память под статус извлечения
  CUDA_CHECK(cudaMalloc(&d_deq_out,n_ops * sizeof(int))); // Память под извлеченные значения

  int threads = 256; // Количество потоков в блоке
  int blocks  = (n_ops + threads - 1) / threads; // Вычисление количества блоков сетки

  // enqueue
  queue_enqueue_kernel<<<blocks, threads>>>(d_buf, d_head, d_tail, cap, n_ops, d_enq_ok); // Заполнение очереди
  CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения операции

  // dequeue
  queue_dequeue_kernel<<<blocks, threads>>>(d_buf, d_head, d_tail, cap,
                                            cap, d_deq_out, d_deq_ok); // Опорожнение очереди
  CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения операции

  std::vector<int> h_enq_ok(n_ops), h_deq_ok(cap), h_deq_out(cap); // Буферы на CPU
  CUDA_CHECK(cudaMemcpy(h_enq_ok.data(), d_enq_ok, n_ops*sizeof(int), cudaMemcpyDeviceToHost)); // Копирование статусов enqueue
  CUDA_CHECK(cudaMemcpy(h_deq_ok.data(), d_deq_ok, cap*sizeof(int), cudaMemcpyDeviceToHost)); // Копирование статусов dequeue
  CUDA_CHECK(cudaMemcpy(h_deq_out.data(),d_deq_out,cap*sizeof(int), cudaMemcpyDeviceToHost)); // Копирование данных dequeue

  verify_queue(h_enq_ok, h_deq_ok, h_deq_out, n_ops); // Проверка корректности работы

  float enq_ms = measure_enqueue(d_buf, d_head, d_tail, cap,
                                 n_ops, d_enq_ok, blocks, threads); // Замер скорости enqueue

  std::cout << "\nПроизводительность:\n"; // Вывод заголовка
  std::cout << "  enqueue kernel (среднее): "
            << std::fixed << std::setprecision(4) << enq_ms << " мс\n"; // Печать результата замера

  CUDA_CHECK(cudaFree(d_buf)); // Освобождение буфера на GPU
  CUDA_CHECK(cudaFree(d_head)); // Освобождение головы на GPU
  CUDA_CHECK(cudaFree(d_tail)); // Освобождение хвоста на GPU
  CUDA_CHECK(cudaFree(d_enq_ok)); // Освобождение массива статусов enqueue
  CUDA_CHECK(cudaFree(d_deq_ok)); // Освобождение массива статусов dequeue
  CUDA_CHECK(cudaFree(d_deq_out)); // Освобождение массива данных dequeue
  return 0; // Завершение программы
} // Конец функции main