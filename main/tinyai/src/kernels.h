/* kernels.h - the numeric hot spots of tinyai, in C and in assembly.
 *
 * A neural network spends nearly all of its time in two loops:
 *
 *     dot : s    = sum_i a[i] * b[i]        (forward pass: one per neuron)
 *     axpy: y[i] += alpha * x[i]            (backward pass + weight update)
 *
 * Each exists twice: a plain C version (the reference, and the fallback for
 * CPUs without AVX2) and a hand-written AVX2+FMA version in asm/kernels.asm.
 * The `ai_backend` table lets the rest of the code stay agnostic.
 */
#ifndef TINYAI_KERNELS_H
#define TINYAI_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Assembly (asm/kernels.asm, Microsoft x64 ABI) -------------------------
 * Only call the *_avx2 kernels when ai_cpu_has_avx2_fma() returned 1.
 * All counts are size_t on purpose: the upper half of an `int` argument
 * register is undefined under the Win64 ABI, so an int count would be a bug. */
unsigned ai_cpu_has_avx2_fma(void);
float    ai_dot_avx2 (const float *a, const float *b, size_t n);
void     ai_axpy_avx2(size_t n, float alpha, const float *x, float *y);

/* ---- Portable C reference --------------------------------------------------*/
float    ai_dot_scalar (const float *a, const float *b, size_t n);
void     ai_axpy_scalar(size_t n, float alpha, const float *x, float *y);

typedef float (*ai_dot_fn )(const float *a, const float *b, size_t n);
typedef void  (*ai_axpy_fn)(size_t n, float alpha, const float *x, float *y);

typedef struct ai_backend {
    const char *name;
    ai_dot_fn   dot;
    ai_axpy_fn  axpy;
} ai_backend;

extern const ai_backend ai_backend_scalar;   /* plain C            */
extern const ai_backend ai_backend_avx2;     /* NASM, AVX2 + FMA   */

/* Best backend this machine can run.  Returns the scalar backend if
 * `force_scalar` is non-zero, if the environment variable AI_FORCE_SCALAR is
 * set to anything other than "" or "0", or if the CPU/OS lacks AVX2+FMA. */
const ai_backend *ai_select_backend(int force_scalar);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_KERNELS_H */
