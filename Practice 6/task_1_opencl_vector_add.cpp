// task_1_opencl_vector_add.cpp
#include <CL/cl.h>

#include <chrono>
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
    if (!f) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static const char* device_type_name(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU) return "GPU";
    if (t & CL_DEVICE_TYPE_CPU) return "CPU";
    return "OTHER";
}

static std::string get_device_string(cl_device_id dev, cl_device_info param) {
    size_t sz = 0;
    clGetDeviceInfo(dev, param, 0, nullptr, &sz);
    std::string s(sz, '\0');
    clGetDeviceInfo(dev, param, sz, s.data(), nullptr);
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static bool run_on_device_type(cl_device_type dtype,
                               int n,
                               int iters,
                               double& out_ms_avg,
                               std::string& out_device_name)
{
    cl_int err;

    // --- platform ---
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    check(err, "clGetPlatformIDs(count)");
    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    check(err, "clGetPlatformIDs(list)");

    // --- pick first platform that has device type ---
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;

    for (auto p : platforms) {
        cl_uint num_devs = 0;
        err = clGetDeviceIDs(p, dtype, 0, nullptr, &num_devs);
        if (err == CL_DEVICE_NOT_FOUND || num_devs == 0) continue;
        check(err, "clGetDeviceIDs(count)");
        std::vector<cl_device_id> devs(num_devs);
        err = clGetDeviceIDs(p, dtype, num_devs, devs.data(), nullptr);
        check(err, "clGetDeviceIDs(list)");
        platform = p;
        device = devs[0]; // берем первый
        break;
    }

    if (!platform || !device) {
        std::cerr << "No OpenCL device found for type: " << device_type_name(dtype) << "\n";
        return false;
    }

    out_device_name = get_device_string(device, CL_DEVICE_NAME);

    // --- context + queue ---
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

#if defined(CL_VERSION_2_0)
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    check(err, "clCreateCommandQueueWithProperties");
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");
#endif

    // --- program build ---
    std::string src = read_text_file("task_1_kernel.cl");
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();

    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // print build log
        size_t log_sz = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log << "\n";
        std::exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    check(err, "clCreateKernel");

    // --- host data ---
    std::vector<float> A(n), B(n), C(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        A[i] = 1.0f + 0.001f * float(i);
        B[i] = 2.0f - 0.0005f * float(i);
    }

    // --- buffers ---
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * n, nullptr, &err);
    check(err, "clCreateBuffer(A)");
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * n, nullptr, &err);
    check(err, "clCreateBuffer(B)");
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &err);
    check(err, "clCreateBuffer(C)");

    // write inputs
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(float) * n, A.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueWriteBuffer(A)");
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(float) * n, B.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueWriteBuffer(B)");

    // --- kernel args ---
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    check(err, "clSetKernelArg(0)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    check(err, "clSetKernelArg(1)");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    check(err, "clSetKernelArg(2)");
    // err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    // check(err, "clSetKernelArg(3)");

    // --- warmup ---
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    check(err, "clEnqueueNDRangeKernel(warmup)");
    clFinish(queue);

    // --- timing: average over iters (kernel only) ---
    using clock = std::chrono::high_resolution_clock;
    double sum_ms = 0.0;

    for (int t = 0; t < iters; ++t) {
        auto t0 = clock::now();
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        check(err, "clEnqueueNDRangeKernel(run)");
        clFinish(queue);
        auto t1 = clock::now();
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        sum_ms += dt.count();
    }

    out_ms_avg = sum_ms / double(iters);

    // read back (and do a small correctness check)
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * n, C.data(), 0, nullptr, nullptr);
    check(err, "clEnqueueReadBuffer(C)");

    // verify few points
    for (int i : {0, n / 2, n - 1}) {
        float ref = A[i] + B[i];
        float diff = std::abs(C[i] - ref);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << C[i] << ", ref " << ref << "\n";
            std::exit(1);
        }
    }

    // cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return true;
}

int main(int argc, char** argv) {
    int n = 1 << 25;     // ~4 million
    int iters = 20;

    if (argc >= 2) n = std::stoi(argv[1]);
    if (argc >= 3) iters = std::stoi(argv[2]);

    std::cout << "Task 1: OpenCL vector_add\n";
    std::cout << "n = " << n << ", iters = " << iters << "\n\n";

    double cpu_ms = 0.0, gpu_ms = 0.0;
    std::string cpu_name, gpu_name;

    bool has_cpu = run_on_device_type(CL_DEVICE_TYPE_CPU, n, iters, cpu_ms, cpu_name);
    bool has_gpu = run_on_device_type(CL_DEVICE_TYPE_GPU, n, iters, gpu_ms, gpu_name);

    std::cout << std::fixed << std::setprecision(4);

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
        double speedup = cpu_ms / gpu_ms;
        std::cout << "Speedup CPU/GPU: " << speedup << "x\n";
    }

    // CSV for graph
    std::ofstream csv("task_1_results.csv");
    csv << "device,ms_avg\n";
    if (has_cpu) csv << "CPU," << cpu_ms << "\n";
    if (has_gpu) csv << "GPU," << gpu_ms << "\n";
    csv.close();

    std::cout << "Saved: task_1_results.csv\n";

    std::ofstream gp("task_1_plot.gp");
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

    system("gnuplot task_1_plot.gp");


    std::cout << "Done.\n";
    return 0;
}
