// task_2_opencl_matmul.cpp                          // Имя файла: OpenCL реализация умножения матриц
#include <CL/cl.h>                                  // Основной заголовок OpenCL API

#include <chrono>                                   // Для измерения времени выполнения
#include <cmath>                                    // Для математических функций (fabs)
#include <fstream>                                  // Для работы с файлами
#include <iomanip>                                  // Для форматирования вывода
#include <iostream>                                 // Для стандартного ввода-вывода
#include <sstream>                                  // Для строковых потоков
#include <string>                                   // Для std::string
#include <vector>                                   // Для std::vector

static void check(cl_int err, const char* what) {   // Функция проверки ошибок OpenCL
    if (err != CL_SUCCESS) {                         // Если код ошибки не равен CL_SUCCESS
        std::cerr << "OpenCL error " << err          // Выводим код ошибки
                  << " at: " << what << "\n";        // Указываем место возникновения
        std::exit(1);                                // Немедленно завершаем программу
    }
}

static std::string read_text_file(const std::string& path) { // Чтение текстового файла в строку
    std::ifstream f(path, std::ios::binary);         // Открываем файл в бинарном режиме
    if (!f) throw std::runtime_error(                // Если файл не открылся
        "Cannot open file: " + path);                // Генерируем исключение
    std::ostringstream ss;                           // Создаем строковый поток
    ss << f.rdbuf();                                 // Считываем весь файл
    return ss.str();                                 // Возвращаем содержимое файла
}

static std::string get_device_string(cl_device_id dev, cl_device_info param) { // Получение строки с параметрами устройства
    size_t sz = 0;                                  // Переменная для хранения размера
    clGetDeviceInfo(dev, param, 0, nullptr, &sz);   // Запрашиваем необходимый размер
    std::string s(sz, '\0');                        // Создаем строку нужного размера
    clGetDeviceInfo(dev, param, sz, s.data(), nullptr); // Получаем строковые данные
    while (!s.empty() &&                            // Удаляем лишние завершающие символы
          (s.back() == '\0' || s.back() == '\n' || s.back() == '\r'))
        s.pop_back();                               // Удаляем последний символ
    return s;                                       // Возвращаем очищенную строку
}

static void cpu_matmul(const std::vector<float>& A, // Последовательное CPU-умножение матриц
                       const std::vector<float>& B, // Матрица B
                       std::vector<float>& C,       // Результирующая матрица C
                       int N, int M, int K)          // Размеры матриц
{
    // C(NxK) = A(NxM) * B(MxK)                        // Формула умножения матриц
    for (int r = 0; r < N; ++r) {                    // Проход по строкам A
        for (int c = 0; c < K; ++c) {                // Проход по столбцам B
            double acc = 0.0;                        // Аккумулятор суммы
            for (int i = 0; i < M; ++i) {            // Суммирование произведений
                acc += double(A[r * M + i])          // Элемент строки A
                     * double(B[i * K + c]);         // Элемент столбца B
            }
            C[r * K + c] = float(acc);               // Записываем результат
        }
    }
}

static size_t round_up(size_t x, size_t m) {         // Округление вверх до кратного m
    return (x + m - 1) / m * m;                      // Формула округления
}

int main(int argc, char** argv) {                    // Точка входа программы
    // Размеры по умолчанию
    int N = 512, M = 512, K = 512;                   // Размеры матриц
    int iters = 10;                                  // Количество итераций

    // argv: N M K iters
    if (argc >= 2) N = std::stoi(argv[1]);           // Читаем N из аргументов
    if (argc >= 3) M = std::stoi(argv[2]);           // Читаем M
    if (argc >= 4) K = std::stoi(argv[3]);           // Читаем K
    if (argc >= 5) iters = std::stoi(argv[4]);       // Читаем число итераций

    std::cout << "Task 2: OpenCL matrix multiplication (C = A[NxM] * B[MxK])\n"; // Заголовок
    std::cout << "N=" << N << " M=" << M << " K=" << K << " iters=" << iters << "\n\n"; // Параметры

    // ---------- OpenCL: platform/device ----------
    cl_int err;                                     // Переменная для кодов ошибок
    cl_uint num_platforms = 0;                      // Количество платформ
    err = clGetPlatformIDs(0, nullptr, &num_platforms); // Запрос количества платформ
    check(err, "clGetPlatformIDs(count)");           // Проверка ошибки
    if (num_platforms == 0) {                        // Если платформ нет
        std::cerr << "No OpenCL platforms found.\n"; // Сообщаем об ошибке
        return 1;                                    // Завершаем программу
    }
    std::vector<cl_platform_id> platforms(num_platforms); // Вектор платформ
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr); // Получаем платформы
    check(err, "clGetPlatformIDs(list)");            // Проверка ошибки

    // Выбор устройства: предпочитаем GPU, если есть
    cl_device_id device = nullptr;                   // Идентификатор устройства
    cl_platform_id platform = nullptr;               // Идентификатор платформы

    auto try_pick = [&](cl_device_type dtype) -> bool { // Лямбда для выбора устройства
        for (auto p : platforms) {                   // Проходим по платформам
            cl_uint num_devs = 0;                    // Количество устройств
            cl_int e = clGetDeviceIDs(p, dtype, 0, nullptr, &num_devs); // Проверка наличия устройств
            if (e == CL_DEVICE_NOT_FOUND || num_devs == 0) continue; // Если нет — пропускаем
            check(e, "clGetDeviceIDs(count)");        // Проверка ошибки
            std::vector<cl_device_id> devs(num_devs); // Вектор устройств
            e = clGetDeviceIDs(p, dtype, num_devs, devs.data(), nullptr); // Получаем устройства
            check(e, "clGetDeviceIDs(list)");         // Проверка ошибки
            platform = p;                            // Сохраняем платформу
            device = devs[0];                        // Берем первое устройство
            return true;                             // Устройство найдено
        }
        return false;                                // Устройство не найдено
    };

    bool ok = try_pick(CL_DEVICE_TYPE_GPU);           // Пытаемся выбрать GPU
    if (!ok) ok = try_pick(CL_DEVICE_TYPE_CPU);       // Если нет — выбираем CPU
    if (!ok || !device) {                             // Если устройство не найдено
        std::cerr << "No OpenCL device found.\n";     // Сообщаем об ошибке
        return 1;                                    // Завершаем программу
    }

    std::cout << "OpenCL device: "                    // Вывод информации об устройстве
              << get_device_string(device, CL_DEVICE_NAME) << "\n\n";

    // ---------- context + queue ----------
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); // Создаем контекст
    check(err, "clCreateContext");                    // Проверка ошибки

#if defined(CL_VERSION_2_0)
    cl_command_queue queue =                          // Очередь команд (OpenCL 2.0)
        clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    check(err, "clCreateCommandQueueWithProperties");
#else
    cl_command_queue queue =                          // Очередь команд (OpenCL < 2.0)
        clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");
#endif

    // ---------- build kernel ----------
    std::string src = read_text_file("task_2_kernel.cl"); // Читаем kernel-файл
    const char* src_ptr = src.c_str();                // Указатель на код
    size_t src_len = src.size();                      // Длина исходника

    cl_program program =                              // Создаем программу OpenCL
        clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // Компиляция программы
    if (err != CL_SUCCESS) {                          // Если сборка не удалась
        size_t log_sz = 0;                            // Размер лога
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');                // Буфер для лога
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log << "\n";// Вывод лога
        return 1;                                    // Завершаем программу
    }

    cl_kernel kernel =                                // Создаем ядро
        clCreateKernel(program, "matmul_tiled", &err);
    check(err, "clCreateKernel(matmul_tiled)");

    // ---------- host matrices ----------
    std::vector<float> A(size_t(N) * M);              // Матрица A
    std::vector<float> B(size_t(M) * K);              // Матрица B
    std::vector<float> C(size_t(N) * K, 0.0f);        // Результат OpenCL
    std::vector<float> C_ref(size_t(N) * K, 0.0f);    // Результат CPU

    // детерминированная инициализация
    for (int r = 0; r < N; ++r)                       // Заполнение A
        for (int c = 0; c < M; ++c)
            A[r * M + c] = 0.01f * float((r + c) % 100);

    for (int r = 0; r < M; ++r)                       // Заполнение B
        for (int c = 0; c < K; ++c)
            B[r * K + c] = 0.02f * float((r * 3 + c) % 100);

    // ---------- buffers ----------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * A.size(), nullptr, &err);
    check(err, "clCreateBuffer(A)");
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * B.size(), nullptr, &err);
    check(err, "clCreateBuffer(B)");
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size(), nullptr, &err);
    check(err, "clCreateBuffer(C)");

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(float) * A.size(), A.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueWriteBuffer(A)");
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(float) * B.size(), B.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueWriteBuffer(B)");

    // ---------- args ----------
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA); check(err, "clSetKernelArg(0)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB); check(err, "clSetKernelArg(1)");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC); check(err, "clSetKernelArg(2)");
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);      check(err, "clSetKernelArg(3)");
    err = clSetKernelArg(kernel, 4, sizeof(int), &M);      check(err, "clSetKernelArg(4)");
    err = clSetKernelArg(kernel, 5, sizeof(int), &K);      check(err, "clSetKernelArg(5)");

    // ---------- NDRange ----------
    const size_t TS = 16;                              // Размер тайла
    size_t local[2]  = {TS, TS};                       // Размер локальной группы
    size_t global[2] = {round_up((size_t)K, TS),       // Глобальный размер по X
                        round_up((size_t)N, TS)};     // Глобальный размер по Y

    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    check(err, "clEnqueueNDRangeKernel(warmup)");
    clFinish(queue);                                  // Ожидание завершения

    using clock = std::chrono::high_resolution_clock; // Тип таймера
    double sum_ms = 0.0;                              // Суммарное время

    for (int t = 0; t < iters; ++t) {                 // Цикл измерений
        auto t0 = clock::now();                       // Время начала
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
        check(err, "clEnqueueNDRangeKernel(run)");
        clFinish(queue);                              // Ожидание
        auto t1 = clock::now();                       // Время конца
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        sum_ms += dt.count();                         // Суммирование
    }
    double ocl_ms = sum_ms / double(iters);           // Среднее время

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueReadBuffer(C)");

    auto c0 = clock::now();                           // CPU начало
    cpu_matmul(A, B, C_ref, N, M, K);                 // CPU вычисление
    auto c1 = clock::now();                           // CPU конец
    std::chrono::duration<double, std::milli> cpu_dt = c1 - c0;

    float max_abs_err = 0.0f;                         // Максимальная ошибка
    for (size_t i = 0; i < C.size(); ++i) {
        float e = std::fabs(C[i] - C_ref[i]);
        if (e > max_abs_err) max_abs_err = e;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "OpenCL avg kernel time (ms): " << ocl_ms << "\n";
    std::cout << "CPU reference time (ms):    " << cpu_dt.count() << "\n";
    std::cout << "Max abs error:              " << max_abs_err << "\n";

    clReleaseMemObject(bufA);                         // Освобождение буфера A
    clReleaseMemObject(bufB);                         // Освобождение буфера B
    clReleaseMemObject(bufC);                         // Освобождение буфера C
    clReleaseKernel(kernel);                          // Освобождение ядра
    clReleaseProgram(program);                        // Освобождение программы
    clReleaseCommandQueue(queue);                     // Освобождение очереди
    clReleaseContext(context);                        // Освобождение контекста

    return 0;                                         // Успешное завершение
}
