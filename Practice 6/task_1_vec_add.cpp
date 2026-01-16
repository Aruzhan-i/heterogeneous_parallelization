#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

// Загрузка kernel из файла
std::string loadKernel(const char* name) {
    std::ifstream file(name);
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

int main() {
    const int N = 1 << 20;   // ~1 млн элементов
    size_t bytes = N * sizeof(float);

    // -------------------------
    // 1. Подготовка данных
    // -------------------------
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        A[i] = i * 0.5f;
        B[i] = i * 2.0f;
    }

    // -------------------------
    // 2. Получение платформы
    // -------------------------
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // -------------------------
    // 3. Получение устройства
    // -------------------------
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);

    // -------------------------
    // 4. Контекст и очередь
    // -------------------------
    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // -------------------------
    // 5. Буферы
    // -------------------------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr);

    // -------------------------
    // 6. Загрузка ядра
    // -------------------------
    std::string source = loadKernel("kernel_vec_add.cl");
    const char* src = source.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);

    // -------------------------
    // 7. Аргументы ядра
    // -------------------------
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // -------------------------
    // 8. Запуск и замер времени
    // -------------------------
    size_t globalSize = N;

    auto start = std::chrono::high_resolution_clock::now();

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                           &globalSize, nullptr,
                           0, nullptr, nullptr);
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // -------------------------
    // 9. Чтение результата
    // -------------------------
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr);

    std::cout << "Execution time: " << time_ms << " ms\n";
    std::cout << "Check: C[10] = " << C[10] << std::endl;

    // -------------------------
    // 10. Освобождение памяти
    // -------------------------
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
