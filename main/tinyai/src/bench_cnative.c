/* bench_cnative.c - "what would gcc do on its own?"
 *
 * The same two loops as kernels.c, but this file is compiled with everything
 * gcc offers:  -O3 -march=native -ffast-math -funroll-loops  (see build.ps1),
 * and the pointers are `restrict` (the same no-overlap promise the assembly
 * relies on).  It exists only so bench.exe can show the hand-written assembly
 * next to an auto-vectorised C compiler, instead of only next to a deliberately
 * plain -O2 loop.  The training code never calls these. */
#include <stddef.h>

float bench_dot_cnative(const float *restrict a, const float *restrict b, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

void bench_axpy_cnative(size_t n, float alpha, const float *restrict x, float *restrict y)
{
    for (size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}
