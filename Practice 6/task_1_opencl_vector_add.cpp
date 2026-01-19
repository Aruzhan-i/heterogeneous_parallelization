// task_1_opencl_vector_add.cpp                     // Имя файла с реализацией OpenCL сложения векторов
#include <CL/cl.h>                                 // Подключение основного заголовка OpenCL API

#include <chrono>                                  // Для высокоточного измерения времени
#include <fstream>                                 // Для работы с файлами (ifstream, ofstream)
#include <iomanip>                                 // Для форматирования вывода (setprecision)
#include <iostream>                                // Для ввода-вывода (cout, cerr)
#include <sstream>                                 // Для работы со строковыми потоками
#include <string>                                  // Для std::string
#include <vector>                                  // Для std::vector

static void check(cl_int err, const char* what) {  // Функция проверки кода ошибки OpenCL
    if (err != CL_SUCCESS) {                        // Если код ошибки не равен успешному
        std::cerr << "OpenCL error " << err         // Выводим сообщение об ошибке
                  << " at: " << what << "\n";       // Указываем место возникновения ошибки
        std::exit(1);                               // Завершаем программу с ошибкой
    }
}

static std::string read_text_file(const std::string& path) { // Функция чтения текстового файла
    std::ifstream f(path, std::ios::binary);        // Открываем файл в бинарном режиме
    if (!f) {                                       // Если файл не открылся
        throw std::runtime_error(                  // Генерируем исключение
            "Cannot open file: " + path);           // С сообщением об ошибке
    }
    std::ostringstream ss;                          // Создаем строковый поток
    ss << f.rdbuf();                                // Читаем весь файл в поток
    return ss.str();                                // Возвращаем содержимое файла строкой
}

static const char* device_type_name(cl_device_type t) { // Определение типа устройства
    if (t & CL_DEVICE_TYPE_GPU) return "GPU";       // Если GPU — возвращаем "GPU"
    if (t & CL_DEVICE_TYPE_CPU) return "CPU";       // Если CPU — возвращаем "CPU"
    return "OTHER";                                 // Иначе — "OTHER"
}

static std::string get_device_string(cl_device_id dev, cl_device_info param) { // Получение строки с параметром устройства
    size_t sz = 0;                                 // Переменная для размера строки
    clGetDeviceInfo(dev, param, 0, nullptr, &sz);  // Узнаем необходимый размер
    std::string s(sz, '\0');                       // Создаем строку нужного размера
    clGetDeviceInfo(dev, param, sz, s.data(), nullptr); // Запрашиваем информацию об устройстве
    while (!s.empty() &&                           // Удаляем завершающие нулевые символы
           (s.back() == '\0' || s.back() == '\n' || s.back() == '\r'))
        s.pop_back();                              // Удаляем последний символ
    return s;                                      // Возвращаем очищенную строку
}

static bool run_on_device_type(cl_device_type dtype, // Функция запуска вычислений на устройстве заданного типа
                               int n,                // Размер вектора
                               int iters,            // Количество итераций для усреднения времени
                               double& out_ms_avg,   // Среднее время выполнения (выход)
                               std::string& out_device_name) // Имя устройства (выход)
{
    cl_int err;                                    // Переменная для кодов ошибок OpenCL

    // --- platform ---                             // Работа с платформами OpenCL
    cl_uint num_platforms = 0;                     // Количество платформ
    err = clGetPlatformIDs(0, nullptr, &num_platforms); // Получаем количество платформ
    check(err, "clGetPlatformIDs(count)");          // Проверяем на ошибку
    if (num_platforms == 0) {                       // Если платформ нет
        std::cerr << "No OpenCL platforms found.\n";// Сообщаем об ошибке
        return false;                               // Возвращаем false
    }
    std::vector<cl_platform_id> platforms(num_platforms); // Вектор платформ
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr); // Получаем список платформ
    check(err, "clGetPlatformIDs(list)");           // Проверяем ошибку

    // --- pick first platform that has device type --- // Выбор платформы с нужным типом устройства
    cl_platform_id platform = nullptr;              // Переменная платформы
    cl_device_id device = nullptr;                  // Переменная устройства

    for (auto p : platforms) {                      // Проходим по всем платформам
        cl_uint num_devs = 0;                       // Количество устройств
        err = clGetDeviceIDs(p, dtype, 0, nullptr, &num_devs); // Проверяем наличие устройств нужного типа
        if (err == CL_DEVICE_NOT_FOUND || num_devs == 0) continue; // Если нет — пропускаем
        check(err, "clGetDeviceIDs(count)");        // Проверяем ошибку
        std::vector<cl_device_id> devs(num_devs);   // Вектор устройств
        err = clGetDeviceIDs(p, dtype, num_devs, devs.data(), nullptr); // Получаем устройства
        check(err, "clGetDeviceIDs(list)");         // Проверяем ошибку
        platform = p;                               // Сохраняем платформу
        device = devs[0];                           // Берем первое устройство
        break;                                      // Выходим из цикла
    }

    if (!platform || !device) {                     // Если устройство не найдено
        std::cerr << "No OpenCL device found for type: "
                  << device_type_name(dtype) << "\n"; // Сообщаем тип устройства
        return false;                               // Возвращаем false
    }

    out_device_name = get_device_string(device, CL_DEVICE_NAME); // Получаем имя устройства

    // --- context + queue ---                      // Создание контекста и очереди команд
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); // Создаем контекст
    check(err, "clCreateContext");                  // Проверяем ошибку

#if defined(CL_VERSION_2_0)
    cl_command_queue queue =                        // Создаем очередь команд (OpenCL 2.0)
        clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    check(err, "clCreateCommandQueueWithProperties"); // Проверяем ошибку
#else
    cl_command_queue queue =                        // Создаем очередь команд (OpenCL < 2.0)
        clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");             // Проверяем ошибку
#endif

    // --- program build ---                        // Сборка OpenCL-программы
    std::string src = read_text_file("task_1_kernel.cl"); // Читаем исходный код ядра
    const char* src_ptr = src.c_str();              // Указатель на строку
    size_t src_len = src.size();                    // Длина строки

    cl_program program =                            // Создаем программу OpenCL
        clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    check(err, "clCreateProgramWithSource");        // Проверяем ошибку

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // Компилируем программу
    if (err != CL_SUCCESS) {                        // Если сборка не удалась
        size_t log_sz = 0;                          // Размер лога
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz); // Получаем размер лога
        std::string log(log_sz, '\0');              // Создаем строку для лога
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr); // Получаем лог
        std::cerr << "Build failed:\n" << log << "\n"; // Выводим лог ошибки
        std::exit(1);                               // Завершаем программу
    }

    cl_kernel kernel =                              // Создаем объект ядра
        clCreateKernel(program, "vector_add", &err);
    check(err, "clCreateKernel");                   // Проверяем ошибку

    // --- host data ---                            // Подготовка данных на хосте
    std::vector<float> A(n), B(n), C(n, 0.0f);      // Векторы A, B и C
    for (int i = 0; i < n; ++i) {                   // Инициализация данных
        A[i] = 1.0f + 0.001f * float(i);            // Заполняем A
        B[i] = 2.0f - 0.0005f * float(i);           // Заполняем B
    }

    // --- buffers ---                              // Создание буферов на устройстве
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * n, nullptr, &err); // Буфер A
    check(err, "clCreateBuffer(A)");                // Проверка ошибки
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(float) * n, nullptr, &err); // Буфер B
    check(err, "clCreateBuffer(B)");                // Проверка ошибки
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * n, nullptr, &err); // Буфер C
    check(err, "clCreateBuffer(C)");                // Проверка ошибки

    // write inputs                                // Запись входных данных в память устройства
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                               sizeof(float) * n, A.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueWriteBuffer(A)");          // Проверка ошибки
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
                               sizeof(float) * n, B.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueWriteBuffer(B)");          // Проверка ошибки

    // --- kernel args ---                          // Установка аргументов ядра
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA); // Аргумент A
    check(err, "clSetKernelArg(0)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB); // Аргумент B
    check(err, "clSetKernelArg(1)");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC); // Аргумент C
    check(err, "clSetKernelArg(2)");
    // err = clSetKernelArg(kernel, 3, sizeof(int), &n);     // Размер (не используется)
    // check(err, "clSetKernelArg(3)");

    // --- warmup ---                              // Прогрев ядра
    size_t global = (size_t)n;                     // Размер глобального диапазона
    err = clEnqueueNDRangeKernel(queue, kernel, 1,
                                 nullptr, &global, nullptr, 0, nullptr, nullptr);
    check(err, "clEnqueueNDRangeKernel(warmup)");
    clFinish(queue);                               // Ожидаем завершения

    // --- timing: average over iters (kernel only) --- // Измерение времени выполнения ядра
    using clock = std::chrono::high_resolution_clock; // Тип часов
    double sum_ms = 0.0;                           // Суммарное время

    for (int t = 0; t < iters; ++t) {              // Повторяем iters раз
        auto t0 = clock::now();                    // Время начала
        err = clEnqueueNDRangeKernel(queue, kernel, 1,
                                     nullptr, &global, nullptr, 0, nullptr, nullptr);
        check(err, "clEnqueueNDRangeKernel(run)");
        clFinish(queue);                           // Ждем завершения
        auto t1 = clock::now();                    // Время конца
        std::chrono::duration<double, std::milli> dt = t1 - t0; // Длительность
        sum_ms += dt.count();                      // Суммируем время
    }

    out_ms_avg = sum_ms / double(iters);           // Среднее время выполнения

    // read back (and do a small correctness check) // Чтение результата
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                              sizeof(float) * n, C.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueReadBuffer(C)");

    // verify few points                           // Проверка корректности
    for (int i : {0, n / 2, n - 1}) {               // Проверяем несколько элементов
        float ref = A[i] + B[i];                   // Эталонное значение
        float diff = std::abs(C[i] - ref);         // Разница
        if (diff > 1e-5f) {                        // Если ошибка слишком большая
            std::cerr << "Mismatch at " << i
                      << ": got " << C[i]
                      << ", ref " << ref << "\n";
            std::exit(1);                          // Завершаем программу
        }
    }

    // cleanup                                    // Освобождение ресурсов
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return true;                                   // Успешное завершение
}

int main(int argc, char** argv) {                  // Точка входа в программу
    int n = 1 << 25;                               // Размер вектора (~4 млн)
    int iters = 20;                                // Количество итераций

    if (argc >= 2) n = std::stoi(argv[1]);         // Чтение n из аргументов
    if (argc >= 3) iters = std::stoi(argv[2]);     // Чтение iters из аргументов

    std::cout << "Task 1: OpenCL vector_add\n";    // Заголовок
    std::cout << "n = " << n << ", iters = "
              << iters << "\n\n";                  // Параметры запуска

    double cpu_ms = 0.0, gpu_ms = 0.0;             // Время CPU и GPU
    std::string cpu_name, gpu_name;                // Имена устройств

    bool has_cpu = run_on_device_type(CL_DEVICE_TYPE_CPU,
                                      n, iters, cpu_ms, cpu_name); // Запуск на CPU
    bool has_gpu = run_on_device_type(CL_DEVICE_TYPE_GPU,
                                      n, iters, gpu_ms, gpu_name); // Запуск на GPU

    std::cout << std::fixed << std::setprecision(4); // Форматирование вывода

    if (has_cpu) {
        std::cout << "[CPU] Device: " << cpu_name << "\n";
        std::cout << "[CPU] Avg kernel time (ms): " << cpu_ms << "\n\n";
    } else {
        std::cout << "[CPU] Not found\n\n";
    }

    if (has_gpu) {
        std::cout << "[GPU] Device: " << gpu_name << "\n";
        std::cout << "[GPU] Avg kernel time (ms): " << gpu_ms << "\n\n";
    } else {
        std::cout << "[GPU] Not found\n\n";
    }

    if (has_cpu && has_gpu) {
        double speedup = cpu_ms / gpu_ms;          // Вычисление ускорения
        std::cout << "Speedup CPU/GPU: "
                  << speedup << "x\n";
    }

    // CSV for graph                              // Сохранение CSV
    std::ofstream csv("task_1_results.csv");
    csv << "device,ms_avg\n";
    if (has_cpu) csv << "CPU," << cpu_ms << "\n";
    if (has_gpu) csv << "GPU," << gpu_ms << "\n";
    csv.close();

    std::cout << "Saved: task_1_results.csv\n";

    std::ofstream gp("task_1_plot.gp");            // Создание gnuplot-скрипта
    gp << "set datafile separator ','\n";
    gp << "set terminal pngcairo size 600,400\n";
    gp << "set output 'task_1_performance.png'\n";
    gp << "set title 'OpenCL vector_add: CPU vs GPU'\n";
    gp << "set ylabel 'Average kernel time (ms)'\n";
    gp << "set yrange [0:*]\n";
    gp << "set grid ytics\n";
    gp << "set style data histograms\n";
    gp << "set style fill solid 0.6\n";
    gp << "set boxwidth 0.5\n";
    gp << "unset key\n";
    gp << "plot 'task_1_results.csv' using 2:xtic(1) every ::1 title 'Execution time'\n";
    gp.close();

    system("gnuplot task_1_plot.gp");              // Запуск gnuplot

    std::cout << "Done.\n";                        // Сообщение о завершении
    return 0;                                     // Успешный выход
}
