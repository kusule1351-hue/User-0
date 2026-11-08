/* kernels.c - portable C reference kernels and backend selection. */
#include "kernels.h"

#include <stdlib.h>

/* These two functions are the specification the assembly is tested against.
 * Build flags matter: this file is compiled with plain -O2 (no -march, no
 * -ffast-math), so GCC may not turn it into AVX code or re-associate the sum.
 * That keeps "scalar" honestly scalar, and keeps the runtime dispatch sound. */
float ai_dot_scalar(const float *a, const float *b, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

void ai_axpy_scalar(size_t n, float alpha, const float *x, float *y)
{
    for (size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

const ai_backend ai_backend_scalar = { "scalar C",        ai_dot_scalar, ai_axpy_scalar };
const ai_backend ai_backend_avx2   = { "AVX2+FMA (NASM)", ai_dot_avx2,   ai_axpy_avx2   };

const ai_backend *ai_select_backend(int force_scalar)
{
    const char *env = getenv("AI_FORCE_SCALAR");
    if (env && env[0] != '\0' && env[0] != '0')
        force_scalar = 1;
    if (!force_scalar && ai_cpu_has_avx2_fma())
        return &ai_backend_avx2;
    return &ai_backend_scalar;
}
