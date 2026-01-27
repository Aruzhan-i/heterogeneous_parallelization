// task1_mean_std_mpi.cpp
// Practice 9 - Task 1 (MPI)
// Rank 0 generates N random numbers, distributes with MPI_Scatterv (handles remainder),
// each process computes local sum and sum of squares,
// rank 0 computes mean and standard deviation:
// sigma = sqrt( (1/N) * sum(x_i^2) - ((1/N) * sum(x_i))^2 )
// Includes timing via MPI_Wtime and prints distribution (local_n, displ) per rank.

#include <mpi.h>                 // MPI API
#include <iostream>              // std::cout, std::cerr
#include <vector>                // std::vector
#include <random>                // RNG
#include <cmath>                 // std::sqrt
#include <cstdlib>               // std::atoll

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);      // Initialize MPI runtime

    int rank = 0;                // Rank (ID) of this process
    int size = 1;                // Total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get size

    // -------------------- Parse N --------------------
    long long N = 1000000;       // Default array size
    if (argc >= 2) {             // If user provided N
        N = std::atoll(argv[1]); // Parse N from argv
        if (N < 0) N = 0;        // Clamp negative to 0
    }

    // -------------------- Build counts/displs for Scatterv --------------------
    std::vector<int> counts(size, 0); // How many elements each rank receives
    std::vector<int> displs(size, 0); // Displacements (start indices) for each rank

    long long base = (size > 0) ? (N / size) : 0; // Base chunk size
    long long rem  = (size > 0) ? (N % size) : 0; // Remainder elements

    for (int p = 0; p < size; ++p) {              // For each process
        long long cnt = base + (p < rem ? 1 : 0); // First 'rem' ranks get +1
        counts[p] = static_cast<int>(cnt);        // Store count (assumes N fits int chunks)
    }

    displs[0] = 0;                                 // First displacement is 0
    for (int p = 1; p < size; ++p) {               // Build prefix sums for displacements
        displs[p] = displs[p - 1] + counts[p - 1]; // displ[p] = sum_{k<p} counts[k]
    }

    int local_n = counts[rank];                    // Local chunk length for this rank
    std::vector<double> local(local_n);            // Local buffer for received data

    // -------------------- Rank 0 generates data --------------------
    std::vector<double> data;                      // Full data exists only on rank 0
    if (rank == 0) {                               // Only rank 0 generates
        data.resize(static_cast<size_t>(N));       // Allocate N elements
        std::mt19937_64 rng(12345);                // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dist(0.0, 1.0); // Random in [0,1)

        for (long long i = 0; i < N; ++i) {        // Fill array
            data[static_cast<size_t>(i)] = dist(rng); // Random value
        }
    }

    // -------------------- Timing variables + start total timer --------------------
    double t0_total = 0.0, t1_total = 0.0;         // Total time
    double t_scatter0 = 0.0, t_scatter1 = 0.0;     // Scatter time
    double t_comp0 = 0.0, t_comp1 = 0.0;           // Compute time
    double t_red0 = 0.0, t_red1 = 0.0;             // Reduce time

    MPI_Barrier(MPI_COMM_WORLD);                   // Sync before timing
    t0_total = MPI_Wtime();                        // Start total timer

    // -------------------- Scatterv timing --------------------
    MPI_Barrier(MPI_COMM_WORLD);                   // Sync before Scatterv timing
    t_scatter0 = MPI_Wtime();                      // Start Scatterv timer

    MPI_Scatterv(
        rank == 0 ? data.data() : nullptr,         // send buffer only on rank 0
        counts.data(),                             // send counts per rank
        displs.data(),                             // displacements per rank
        MPI_DOUBLE,                                // datatype
        local.data(),                              // receive buffer (local chunk)
        local_n,                                   // receive count
        MPI_DOUBLE,                                // datatype
        0,                                         // root rank
        MPI_COMM_WORLD                             // communicator
    );

    MPI_Barrier(MPI_COMM_WORLD);                   // Sync after Scatterv
    t_scatter1 = MPI_Wtime();                      // End Scatterv timer

    // -------------------- Print distribution (rank order) --------------------
    for (int p = 0; p < size; ++p) {               // For ordered printing
        if (rank == p) {                           // Only the current rank prints
            std::cout << "rank " << rank
                      << " got local_n = " << local_n
                      << " (displ = " << displs[rank] << ")\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);               // Ensure ordering
    }

    // -------------------- Local compute timing --------------------
    MPI_Barrier(MPI_COMM_WORLD);                   // Sync before compute timing
    t_comp0 = MPI_Wtime();                         // Start compute timer

    double local_sum = 0.0;                        // Local sum
    double local_sum_sq = 0.0;                     // Local sum of squares
    for (int i = 0; i < local_n; ++i) {            // Iterate local chunk
        local_sum += local[i];                     // Add to sum
        local_sum_sq += local[i] * local[i];       // Add to sum of squares
    }

    MPI_Barrier(MPI_COMM_WORLD);                   // Sync after compute
    t_comp1 = MPI_Wtime();                         // End compute timer

    // -------------------- Reduce timing --------------------
    MPI_Barrier(MPI_COMM_WORLD);                   // Sync before reduce timing
    t_red0 = MPI_Wtime();                          // Start reduce timer

    double global_sum = 0.0;                       // Global sum (root)
    double global_sum_sq = 0.0;                    // Global sum of squares (root)

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);     // Reduce sums
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce sumsq

    MPI_Barrier(MPI_COMM_WORLD);                   // Sync after reduce
    t_red1 = MPI_Wtime();                          // End reduce timer

    // -------------------- Rank 0 computes mean and std --------------------
    if (rank == 0) {                               // Only root prints results
        if (N == 0) {                              // Edge case: empty array
            std::cout << "N = 0, mean/std are undefined.\n"; // Inform user
        } else {
            double mean = global_sum / static_cast<double>(N); // E[x]
            double ex2  = global_sum_sq / static_cast<double>(N); // E[x^2]
            double var  = ex2 - mean * mean;       // Var = E[x^2] - (E[x])^2

            if (var < 0.0 && var > -1e-12) var = 0.0; // Fix tiny negative from rounding

            double sigma = std::sqrt(var);         // Std deviation

            std::cout << "N = " << N << "\n";      // Print N
            std::cout << "Processes = " << size << "\n"; // Print P
            std::cout << "Sum = " << global_sum << "\n"; // Print sum
            std::cout << "SumSq = " << global_sum_sq << "\n"; // Print sumsq
            std::cout << "Mean = " << mean << "\n"; // Print mean
            std::cout << "StdDev = " << sigma << "\n"; // Print stddev
        }
    }

    // -------------------- Stop total timer --------------------
    MPI_Barrier(MPI_COMM_WORLD);                   // Sync before total end
    t1_total = MPI_Wtime();                        // End total timer

    // -------------------- Collect max timings across ranks --------------------
    double scatter_time = t_scatter1 - t_scatter0; // Local scatter duration
    double compute_time = t_comp1 - t_comp0;       // Local compute duration
    double reduce_time  = t_red1 - t_red0;         // Local reduce duration
    double total_time   = t1_total - t0_total;     // Local total duration

    double scatter_max = 0.0, compute_max = 0.0, reduce_max = 0.0, total_max = 0.0; // Max timers on root

    MPI_Reduce(&scatter_time, &scatter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Max scatter
    MPI_Reduce(&compute_time, &compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Max compute
    MPI_Reduce(&reduce_time,  &reduce_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Max reduce
    MPI_Reduce(&total_time,   &total_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Max total

    // -------------------- Print timing on root --------------------
    if (rank == 0) {                               // Only root prints timing summary
        std::cout << "Timing (max over ranks):\n"; // Header
        std::cout << "  Scatterv: " << scatter_max * 1000.0 << " ms\n"; // Scatter time
        std::cout << "  Compute : " << compute_max  * 1000.0 << " ms\n"; // Compute time
        std::cout << "  Reduce  : " << reduce_max   * 1000.0 << " ms\n"; // Reduce time
        std::cout << "  Total   : " << total_max    * 1000.0 << " ms\n"; // Total time
    }

    MPI_Finalize();                                // Finalize MPI runtime
    return 0;                                      // Exit
}
