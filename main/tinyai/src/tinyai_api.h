/* tinyai_api.h - a flat C interface to tinyai for OTHER languages.
 *
 * Built into build\tinyai.dll.  The Common Lisp front end (lisp\tinyai.lisp)
 * loads it through SBCL's FFI, but anything with a C FFI can use it.
 *
 * Conventions
 *   * Everything is a plain C type with an explicit width (uint64_t, never
 *     size_t/int), so the foreign side can declare it without guessing.
 *   * Networks and datasets are opaque handles; free each one exactly once.
 *   * Functions that can fail return NULL (constructors) or NaN (numeric
 *     results) instead of crashing, and never write through a bad handle.
 *   * A handle must not be used from two threads at once.
 */
#ifndef TINYAI_API_H
#define TINYAI_API_H

#include <stdint.h>

#if defined(_WIN32) && defined(TAI_BUILD_DLL)
#  define TAI_API __declspec(dllexport)
#else
#  define TAI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tai_net  tai_net;
typedef struct tai_data tai_data;

/* ---- the CPU and the raw kernels --------------------------------------------- */
/* 1 if AVX2+FMA can be used (CPU *and* OS support), else 0. */
TAI_API uint32_t    tai_cpu_has_avx2_fma(void);
/* Name of the kernels tai_dot/tai_axpy use: "AVX2+FMA (NASM)" or "scalar C". */
TAI_API const char *tai_kernel_name(void);
/* sum a[i]*b[i], y[i] += alpha*x[i].  The best kernels available (the assembly
 * when the CPU has AVX2+FMA) ... */
TAI_API float       tai_dot (const float *a, const float *b, uint64_t n);
TAI_API void        tai_axpy(uint64_t n, float alpha, const float *x, float *y);
/* ... and the plain-C reference versions, for comparison. */
TAI_API float       tai_dot_scalar (const float *a, const float *b, uint64_t n);
TAI_API void        tai_axpy_scalar(uint64_t n, float alpha, const float *x, float *y);

/* ---- datasets ------------------------------------------------------------------ */
/* `classes` interleaved spiral arms, `per_class` points each; see data.h.
 * Returns NULL on bad arguments or out of memory. */
TAI_API tai_data      *tai_spirals_new(uint64_t per_class, uint64_t classes,
                                       float turns, float noise, uint64_t seed);
TAI_API void           tai_data_free(tai_data *d);
TAI_API uint64_t       tai_data_size(const tai_data *d);      /* number of samples   */
TAI_API uint64_t       tai_data_dim(const tai_data *d);       /* features per sample */
TAI_API uint64_t       tai_data_classes(const tai_data *d);
TAI_API const float   *tai_data_x(const tai_data *d);         /* size*dim floats     */
TAI_API const uint32_t*tai_data_y(const tai_data *d);         /* size labels         */

/* ---- networks --------------------------------------------------------------------
 * dims[0..n_layers] are the layer widths, e.g. {2, 64, 64, 3} with n_layers = 3
 * (2 inputs, two hidden ReLU layers of 64, 3 classes).  `seed` fixes the initial
 * weights and the shuffling order, so a run is reproducible.  Returns NULL on bad
 * arguments (n_layers must be 1..8, every width 1..16777216) or out of memory. */
TAI_API tai_net       *tai_net_new(const uint64_t *dims, uint64_t n_layers,
                                   uint64_t seed, int32_t force_scalar);
TAI_API void           tai_net_free(tai_net *n);
TAI_API uint64_t       tai_net_n_params(const tai_net *n);
TAI_API uint64_t       tai_net_inputs(const tai_net *n);
TAI_API uint64_t       tai_net_outputs(const tai_net *n);
TAI_API const char    *tai_net_backend(const tai_net *n);

/* One SGD pass over the dataset (shuffled).  Returns the mean training loss, or
 * NaN if the dataset does not fit the network (wrong feature count, or more
 * classes than the network has outputs). */
TAI_API double         tai_net_train_epoch(tai_net *n, const tai_data *d,
                                           float lr, uint64_t batch);
/* Mean loss and accuracy (percent) on a dataset; both set to NaN on mismatch. */
TAI_API void           tai_net_evaluate(tai_net *n, const tai_data *d,
                                        double *loss, double *acc_percent);
/* x must hold tai_net_inputs(n) floats.  logits() returns tai_net_outputs(n)
 * floats owned by the network (valid until its next call). */
TAI_API uint64_t       tai_net_predict(tai_net *n, const float *x);
TAI_API const float   *tai_net_logits (tai_net *n, const float *x);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_API_H */
