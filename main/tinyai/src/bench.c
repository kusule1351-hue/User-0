/* bench.c - how much does the assembly actually buy?  Measured, not assumed.
 *
 * Part 1: the two kernels in isolation, at several vector lengths, against
 *           * scalar C            gcc -O2                       (the fallback)
 *           * auto-vectorised C   gcc -O3 -march=native -ffast-math -funroll-loops
 *                                 (what the compiler can do by itself)
 *           * the NASM AVX2+FMA kernels
 * Part 2: end-to-end training speed (forward + backward + SGD step per sample),
 *         scalar vs AVX2, on three network sizes.
 *
 * How the numbers are kept honest (each of these fixed a real problem):
 *   * The thread is pinned to a performance core.  On a hybrid P/E-core CPU an
 *     un-pinned run can be silently moved to an efficiency core mid-measurement.
 *   * The variants are timed ROUND-ROBIN, not one after the other, and each one
 *     keeps its best round.  A slow patch (migration, power limit) then hits all
 *     contenders alike instead of wrecking whichever one happened to be running.
 *   * Buffers are 64-byte aligned.  A 32-byte vector access that straddles a
 *     cache line costs about 40% throughput (measured: 44-50 vs 78-80 GFLOP/s for
 *     axpy at n=1024), for the assembly and for gcc alike.
 * Data stays in L1/L2 up to n = 16384; n = 1M floats (4 MB per vector) is
 * deliberately memory-bound. */
#include <cpuid.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <malloc.h>
#include <windows.h>

#include "kernels.h"
#include "mlp.h"
#include "rng.h"
#include "timer.h"

/* bench_cnative.c */
float bench_dot_cnative(const float *a, const float *b, size_t n);
void  bench_axpy_cnative(size_t n, float alpha, const float *x, float *y);

#define ROUNDS       11
#define TARGET_FLOPS 2.0e8          /* work per timed block */

static volatile float g_sink;       /* keeps the compiler from deleting the calls */

/* ---- run on a performance core ------------------------------------------------ */
static void pin_to_fastest_core(void)
{
    ULONG len = 0;
    GetSystemCpuSetInformation(NULL, 0, &len, GetCurrentProcess(), 0);
    unsigned char *buf = len ? malloc(len) : NULL;
    if (!buf || !GetSystemCpuSetInformation((PSYSTEM_CPU_SET_INFORMATION)buf, len, &len,
                                            GetCurrentProcess(), 0)) {
        free(buf);
        printf("pinning : unavailable (running un-pinned)\n");
        return;
    }
    int top = -1, bottom = 1 << 20, cpu = -1;
    for (ULONG off = 0; off < len;) {
        const PSYSTEM_CPU_SET_INFORMATION info = (PSYSTEM_CPU_SET_INFORMATION)(buf + off);
        if (info->Type == CpuSetInformation) {
            const int ec = info->CpuSet.EfficiencyClass;      /* higher = faster core */
            if (ec > top && info->CpuSet.Group == 0 && info->CpuSet.LogicalProcessorIndex < 64) {
                top = ec;
                cpu = info->CpuSet.LogicalProcessorIndex;
            }
            if (ec < bottom)
                bottom = ec;
        }
        off += info->Size;
    }
    free(buf);
    if (cpu >= 0 && SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu))
        printf("pinning : logical CPU %d (efficiency class %d of %d..%d)\n", cpu, top, bottom, top);
    else
        printf("pinning : failed (running un-pinned)\n");
}

/* ---- timed blocks: seconds for `reps` calls --------------------------------- */
static double dot_block(ai_dot_fn fn, const float *a, const float *b, size_t n, size_t reps)
{
    float sink = 0.0f;
    const double t0 = ai_now();
    for (size_t r = 0; r < reps; ++r)
        sink += fn(a, b, n);
    const double dt = ai_now() - t0;
    g_sink = sink;
    return dt;
}

static double axpy_block(ai_axpy_fn fn, const float *x, float *y, size_t n, size_t reps)
{
    const double t0 = ai_now();
    for (size_t r = 0; r < reps; ++r)
        fn(n, (r & 1) ? -1e-3f : 1e-3f, x, y);          /* +/- keeps y bounded */
    const double dt = ai_now() - t0;
    g_sink = y[0];
    return dt;
}

static void spin_up(void)                                /* let the clocks ramp up */
{
    const double t0 = ai_now();
    volatile double x = 1.0;
    while (ai_now() - t0 < 0.3)
        x = x * 1.0000001 + 1e-9;
}

static void kernel_table(const char *title, int is_axpy, const float *a, const float *b, float *y)
{
    static const size_t sizes[] = { 16, 64, 256, 1024, 4096, 16384, 1048576 };
    const int have_avx2 = (int)ai_cpu_has_avx2_fma();
    const ai_dot_fn  dot_fns[3]  = { ai_dot_scalar,  bench_dot_cnative,  ai_dot_avx2  };
    const ai_axpy_fn axpy_fns[3] = { ai_axpy_scalar, bench_axpy_cnative, ai_axpy_avx2 };
    const int nv = have_avx2 ? 3 : 2;

    printf("\n%s     GFLOP/s (2 flops per element), best of %d round-robin rounds\n", title, ROUNDS);
    printf("%10s %10s %12s %10s %9s %9s\n", "n", "scalar C", "gcc -O3 vec", "AVX2 asm", "asm/C", "asm/gcc");
    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; ++k) {
        const size_t n = sizes[k];
        size_t reps = (size_t)(TARGET_FLOPS / (2.0 * (double)n));
        if (reps < 2)
            reps = 2;
        double best[3] = { 1e30, 1e30, 1e30 };
        for (int v = 0; v < nv; ++v)                     /* warm-up, one block each */
            (void)(is_axpy ? axpy_block(axpy_fns[v], a, y, n, 2) : dot_block(dot_fns[v], a, b, n, 2));
        for (int r = 0; r < ROUNDS; ++r) {
            for (int v = 0; v < nv; ++v) {
                const double dt = is_axpy ? axpy_block(axpy_fns[v], a, y, n, reps)
                                          : dot_block(dot_fns[v], a, b, n, reps);
                if (dt < best[v])
                    best[v] = dt;
            }
        }
        double g[3] = { 0, 0, 0 };
        for (int v = 0; v < nv; ++v)
            g[v] = 2.0 * (double)n * (double)reps / best[v] * 1e-9;
        if (have_avx2)
            printf("%10zu %10.2f %12.2f %10.2f %8.1fx %8.2fx\n", n, g[0], g[1], g[2], g[2] / g[0], g[2] / g[1]);
        else
            printf("%10zu %10.2f %12.2f %10s %9s %9s\n", n, g[0], g[1], "n/a", "-", "-");
    }
}

/* ---- end to end -------------------------------------------------------------- */
typedef struct net_case {
    ai_mlp  m;
    float  *x;
    size_t  lab[64];
    size_t  nin;
    float   lr;
} net_case;

static int net_case_init(net_case *c, const ai_backend *be, const size_t *dims, size_t layers)
{
    ai_rng rng;
    ai_rng_seed(&rng, 99);
    if (ai_mlp_init(&c->m, dims, layers, be, &rng))
        return -1;
    c->nin = dims[0];
    c->x = malloc(64 * c->nin * sizeof(float));
    if (!c->x)
        return -1;
    for (size_t i = 0; i < 64 * c->nin; ++i)
        c->x[i] = ai_rng_gauss(&rng);
    for (size_t s = 0; s < 64; ++s)
        c->lab[s] = ai_rng_below(&rng, (uint32_t)dims[layers]);
    c->lr = 0.05f / (float)c->nin;                       /* small enough never to diverge */
    return 0;
}

static double net_case_run(net_case *c, size_t steps)
{
    float acc = 0.0f;
    const double t0 = ai_now();
    for (size_t s = 0; s < steps; ++s) {
        const size_t i = s % 64;
        acc += ai_mlp_loss_grad(&c->m, c->x + i * c->nin, c->lab[i]);
        ai_mlp_sgd_step(&c->m, c->lr, 1);
    }
    const double dt = ai_now() - t0;
    g_sink = acc;
    if (!isfinite((double)acc))
        fprintf(stderr, "warning: loss became non-finite during the benchmark\n");
    return dt;
}

static void e2e_table(void)
{
    static const struct {
        const char *name;
        size_t dims[5];
        size_t layers, steps;
    } cases[] = {
        { "2-64-64-3        (the demo)",     { 2, 64, 64, 3 },        3, 60000 },
        { "784-128-10       (MNIST-sized)",  { 784, 128, 10, 0, 0 },  2, 3000  },
        { "784-256-128-10   (bigger)",       { 784, 256, 128, 10 },   3, 1500  },
    };
    const int have_avx2 = (int)ai_cpu_has_avx2_fma();

    printf("\nend-to-end training, one sample per step (forward + backward + SGD update)\n");
    printf("samples/s, best of 5 round-robin rounds\n");
    printf("%-34s %14s %14s %9s\n", "network", "scalar C", "AVX2 asm", "speedup");
    for (size_t k = 0; k < sizeof cases / sizeof *cases; ++k) {
        net_case sc, av;
        if (net_case_init(&sc, &ai_backend_scalar, cases[k].dims, cases[k].layers) ||
            (have_avx2 && net_case_init(&av, &ai_backend_avx2, cases[k].dims, cases[k].layers))) {
            fprintf(stderr, "setup failed\n");
            return;
        }
        double best_sc = 1e30, best_av = 1e30;
        for (int r = 0; r < 5; ++r) {
            const double a = net_case_run(&sc, cases[k].steps);
            if (a < best_sc)
                best_sc = a;
            if (have_avx2) {
                const double b = net_case_run(&av, cases[k].steps);
                if (b < best_av)
                    best_av = b;
            }
        }
        const double rs = (double)cases[k].steps / best_sc;
        if (have_avx2) {
            const double ra = (double)cases[k].steps / best_av;
            printf("%-34s %14.0f %14.0f %8.2fx\n", cases[k].name, rs, ra, ra / rs);
        } else {
            printf("%-34s %14.0f %14s %9s\n", cases[k].name, rs, "n/a", "-");
        }
        ai_mlp_free(&sc.m);
        free(sc.x);
        if (have_avx2) {
            ai_mlp_free(&av.m);
            free(av.x);
        }
    }
}

static void print_cpu(void)
{
    char brand[49] = { 0 };
    if (__get_cpuid_max(0x80000000u, NULL) >= 0x80000004u) {
        unsigned r[12];
        for (unsigned i = 0; i < 3; ++i)
            __get_cpuid(0x80000002u + i, &r[4 * i], &r[4 * i + 1], &r[4 * i + 2], &r[4 * i + 3]);
        memcpy(brand, r, 48);
    }
    const char *p = brand;
    while (*p == ' ')
        ++p;
    printf("CPU     : %s\n", *p ? p : "(unknown)");
    printf("AVX2+FMA: %s\n", ai_cpu_has_avx2_fma() ? "available" : "NOT available");
    printf("builds  : scalar C = gcc -O2 | gcc vec = gcc -O3 -march=native -ffast-math -funroll-loops\n");
    printf("          asm = NASM, AVX2+FMA, 4 independent accumulators, scalar tail\n");
}

int main(void)
{
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    print_cpu();
    pin_to_fastest_core();
    spin_up();

    const size_t max_n = 1048576;
    float *a = _aligned_malloc(max_n * sizeof(float), 64);
    float *b = _aligned_malloc(max_n * sizeof(float), 64);
    float *y = _aligned_malloc(max_n * sizeof(float), 64);
    if (!a || !b || !y) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    ai_rng rng;
    ai_rng_seed(&rng, 5);
    for (size_t i = 0; i < max_n; ++i) {
        a[i] = 2.0f * ai_rng_uniform(&rng) - 1.0f;
        b[i] = 2.0f * ai_rng_uniform(&rng) - 1.0f;
        y[i] = 2.0f * ai_rng_uniform(&rng) - 1.0f;
    }
    kernel_table("dot product   s = sum(a[i]*b[i])", 0, a, b, y);
    kernel_table("axpy          y[i] += alpha*x[i]", 1, a, b, y);
    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(y);

    e2e_table();
    printf("\n(numbers still move a few percent from run to run: turbo clocks, other processes)\n");
    return 0;
}
