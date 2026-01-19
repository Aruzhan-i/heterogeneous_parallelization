// task_2_opencl_matmul.cpp
#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error " << err << " at: " << what << "\n";
        std::exit(1);
    }
}

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string get_device_string(cl_device_id dev, cl_device_info param) {
    size_t sz = 0;
    clGetDeviceInfo(dev, param, 0, nullptr, &sz);
    std::string s(sz, '\0');
    clGetDeviceInfo(dev, param, sz, s.data(), nullptr);
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static void cpu_matmul(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C,
                       int N, int M, int K)
{
    // C(NxK) = A(NxM) * B(MxK)
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < K; ++c) {
            double acc = 0.0;
            for (int i = 0; i < M; ++i) {
                acc += double(A[r * M + i]) * double(B[i * K + c]);
            }
            C[r * K + c] = float(acc);
        }
    }
}

static size_t round_up(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}

int main(int argc, char** argv) {
    // Размеры по умолчанию
    int N = 512, M = 512, K = 512;
    int iters = 10;

    // argv: N M K iters
    if (argc >= 2) N = std::stoi(argv[1]);
    if (argc >= 3) M = std::stoi(argv[2]);
    if (argc >= 4) K = std::stoi(argv[3]);
    if (argc >= 5) iters = std::stoi(argv[4]);

    std::cout << "Task 2: OpenCL matrix multiplication (C = A[NxM] * B[MxK])\n";
    std::cout << "N=" << N << " M=" << M << " K=" << K << " iters=" << iters << "\n\n";

    // ---------- OpenCL: platform/device ----------
    cl_int err;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    check(err, "clGetPlatformIDs(count)");
    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    check(err, "clGetPlatformIDs(list)");

    // Выбор устройства: предпочитаем GPU, если есть
    cl_device_id device = nullptr;
    cl_platform_id platform = nullptr;

    auto try_pick = [&](cl_device_type dtype) -> bool {
        for (auto p : platforms) {
            cl_uint num_devs = 0;
            cl_int e = clGetDeviceIDs(p, dtype, 0, nullptr, &num_devs);
            if (e == CL_DEVICE_NOT_FOUND || num_devs == 0) continue;
            check(e, "clGetDeviceIDs(count)");
            std::vector<cl_device_id> devs(num_devs);
            e = clGetDeviceIDs(p, dtype, num_devs, devs.data(), nullptr);
            check(e, "clGetDeviceIDs(list)");
            platform = p;
            device = devs[0];
            return true;
        }
        return false;
    };

    bool ok = try_pick(CL_DEVICE_TYPE_GPU);
    if (!ok) ok = try_pick(CL_DEVICE_TYPE_CPU);
    if (!ok || !device) {
        std::cerr << "No OpenCL device found.\n";
        return 1;
    }

    std::cout << "OpenCL device: " << get_device_string(device, CL_DEVICE_NAME) << "\n\n";

    // ---------- context + queue ----------
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

#if defined(CL_VERSION_2_0)
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    check(err, "clCreateCommandQueueWithProperties");
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");
#endif

    // ---------- build kernel ----------
    std::string src = read_text_file("task_2_kernel.cl");
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();

    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    check(err, "clCreateProgramWithSource");

    // TS=16 в kernel.cl, но на всякий случай можно переопределять:
    // const char* options = "-DTS=16";
    // err = clBuildProgram(program, 1, &device, options, nullptr, nullptr);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log << "\n";
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matmul_tiled", &err);
    check(err, "clCreateKernel(matmul_tiled)");

    // ---------- host matrices ----------
    std::vector<float> A(size_t(N) * M);
    std::vector<float> B(size_t(M) * K);
    std::vector<float> C(size_t(N) * K, 0.0f);
    std::vector<float> C_ref(size_t(N) * K, 0.0f);

    // детерминированная инициализация
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < M; ++c)
            A[r * M + c] = 0.01f * float((r + c) % 100);

    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            B[r * K + c] = 0.02f * float((r * 3 + c) % 100);

    // ---------- buffers ----------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * A.size(), nullptr, &err);
    check(err, "clCreateBuffer(A)");
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * B.size(), nullptr, &err);
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
    const size_t TS = 16;
    size_t local[2]  = {TS, TS};                // (x=col, y=row)
    size_t global[2] = {round_up((size_t)K, TS),
                        round_up((size_t)N, TS)};

    // warmup
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    check(err, "clEnqueueNDRangeKernel(warmup)");
    clFinish(queue);

    // timing (kernel only)
    using clock = std::chrono::high_resolution_clock;
    double sum_ms = 0.0;

    for (int t = 0; t < iters; ++t) {
        auto t0 = clock::now();
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
        check(err, "clEnqueueNDRangeKernel(run)");
        clFinish(queue);
        auto t1 = clock::now();
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        sum_ms += dt.count();
    }
    double ocl_ms = sum_ms / double(iters);

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueReadBuffer(C)");

    // ---------- correctness vs CPU ----------
    auto c0 = clock::now();
    cpu_matmul(A, B, C_ref, N, M, K);
    auto c1 = clock::now();
    std::chrono::duration<double, std::milli> cpu_dt = c1 - c0;

    float max_abs_err = 0.0f;
    for (size_t i = 0; i < C.size(); ++i) {
        float e = std::fabs(C[i] - C_ref[i]);
        if (e > max_abs_err) max_abs_err = e;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "OpenCL avg kernel time (ms): " << ocl_ms << "\n";
    std::cout << "CPU reference time (ms):    " << cpu_dt.count() << "\n";
    std::cout << "Max abs error:              " << max_abs_err << "\n";

    // cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
