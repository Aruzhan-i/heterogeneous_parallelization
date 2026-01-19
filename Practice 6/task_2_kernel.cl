// task_2_kernel.cl
// C = A(NxM) * B(MxK) => C(NxK)

#define TS 16  // tile size (локальный блок)

__kernel void matmul_tiled(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const int N,
                           const int M,
                           const int K)
{
    const int col = get_global_id(0); // 0..K-1
    const int row = get_global_id(1); // 0..N-1

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    __local float As[TS][TS];
    __local float Bs[TS][TS];

    float acc = 0.0f;

    // Сколько тайлов по M
    const int numTiles = (M + TS - 1) / TS;

    for (int t = 0; t < numTiles; ++t) {
        // Индексы для загрузки тайла
        const int a_col = t * TS + lx;   // по M
        const int b_row = t * TS + ly;   // по M

        // Загрузка A-тайла (row, a_col)
        if (row < N && a_col < M)
            As[ly][lx] = A[row * M + a_col];
        else
            As[ly][lx] = 0.0f;

        // Загрузка B-тайла (b_row, col)
        if (b_row < M && col < K)
            Bs[ly][lx] = B[b_row * K + col];
        else
            Bs[ly][lx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Перемножение внутри тайла
        for (int i = 0; i < TS; ++i)
            acc += As[ly][i] * Bs[i][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < K)
        C[row * K + col] = acc;
}
