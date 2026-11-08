/* test_kernels.c - does the assembly compute what the C reference computes?
 *
 * Every kernel is checked against a DOUBLE-precision reference, so a tolerance
 * only has to absorb float32 rounding.  The suite is first run against the plain
 * C kernels: if the harness itself were wrong (bad tolerance, bad reference) it
 * would fail there before the assembly is ever involved.
 *
 * Checks, per kernel:
 *   * n = 0..70 plus 95..97, 127..129, 255..257, 1003, 4099  (every tail length
 *     of the 32-wide, 8-wide and scalar loops, and several full iterations)
 *   * GUARD PAGES: the buffer ends (or starts) exactly at an unmapped page, so a
 *     single byte read or written outside [ptr, ptr + 4n) crashes the test.
 *     (A "canary float" can only catch overruns that happen to change a value.)
 *   * unaligned pointers, offset by 0..7 floats, inside NaN-poisoned heap blocks
 *   * dot: exact integer data, where any summation order must agree bit-for-bit
 *   * axpy: bytes outside the written window are untouched, x is untouched
 *   * ABI: every Win64 non-volatile register (RBX RBP RDI RSI R12-R15, XMM6-15)
 *     survives each call (asm/abi_check.asm loads canaries and looks afterwards)
 *
 * Exit code 0 = everything passed (or the CPU has no AVX2+FMA, which is reported).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <windows.h>

#include "kernels.h"
#include "rng.h"

static long g_checks, g_fails;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond) && ++g_fails <= 20) {                                      \
            printf("\n    FAIL (line %d): ", __LINE__);                        \
            printf(__VA_ARGS__);                                               \
        }                                                                      \
        else if (!(cond)) { /* counted, not printed */ }                       \
    } while (0)

#define N_SIZES (71 + 11)
static size_t test_size(size_t k)
{
    static const size_t extra[11] = { 95, 96, 97, 127, 128, 129, 255, 256, 257, 1003, 4099 };
    return k < 71 ? k : extra[k - 71];
}

static void fill_random(float *p, size_t n, ai_rng *rng)
{
    for (size_t i = 0; i < n; ++i)
        p[i] = 2.0f * ai_rng_uniform(rng) - 1.0f;
}

/* ---- buffers that sit flush against an unmapped page ------------------------ */
typedef struct guarded {
    unsigned char *base;
    float         *p;
} guarded;

/* [ NOACCESS page ][ n floats' worth of read/write pages ][ NOACCESS page ]
 * at_end = 1: the array ends exactly where the trailing guard page begins.
 * at_end = 0: the array starts exactly where the leading guard page ends.   */
static int guarded_alloc(guarded *g, size_t n, int at_end)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page  = si.dwPageSize;
    const size_t bytes = n * sizeof(float);
    size_t rw = (bytes + page - 1) / page;
    if (rw == 0)
        rw = 1;
    unsigned char *base = VirtualAlloc(NULL, (rw + 2) * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!base)
        return -1;
    DWORD old;
    VirtualProtect(base, page, PAGE_NOACCESS, &old);
    VirtualProtect(base + (rw + 1) * page, page, PAGE_NOACCESS, &old);
    unsigned char *first = base + page;
    g->base = base;
    g->p = (float *)(at_end ? first + rw * page - bytes : first);
    return 0;
}

static void guarded_free(guarded *g)
{
    VirtualFree(g->base, 0, MEM_RELEASE);
}

/* ---- float helpers ----------------------------------------------------------- */
static float make_nan(void)
{
    const uint32_t bits = 0x7FC00000u;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static double dot_ref(const float *a, const float *b, size_t n, double *abs_sum)
{
    double s = 0.0, as = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double p = (double)a[i] * (double)b[i];
        s += p;
        as += fabs(p);
    }
    *abs_sum = as;
    return s;
}

/* ============================== dot =========================================== */

static void test_dot_exact(ai_dot_fn dot)
{
    /* sum of i^2 for i = 1..n stays below 2^24, so every partial sum is exactly
     * representable: ANY association order must give bit-identical results. */
    float a[128];
    for (size_t n = 0; n <= 100; ++n) {
        for (size_t i = 0; i < n; ++i)
            a[i] = (float)(i + 1);
        const float got  = dot(a, a, n);
        const float want = (float)((double)n * (double)(n + 1) * (double)(2 * n + 1) / 6.0);
        CHECK(got == want, "n=%zu got %.9g want %.9g", n, (double)got, (double)want);
    }
}

static void test_dot_guard(ai_dot_fn dot, ai_rng *rng)
{
    for (size_t k = 0; k < N_SIZES; ++k) {
        const size_t n = test_size(k);
        for (int at_end = 0; at_end <= 1; ++at_end) {
            guarded ga, gb;
            if (guarded_alloc(&ga, n, at_end) || guarded_alloc(&gb, n, at_end)) {
                CHECK(0, "VirtualAlloc failed");
                return;
            }
            fill_random(ga.p, n, rng);
            fill_random(gb.p, n, rng);
            double as;
            const double want = dot_ref(ga.p, gb.p, n, &as);
            const float  got  = dot(ga.p, gb.p, n);
            const double tol  = 1e-5 * as + 1e-7;
            CHECK(fabs((double)got - want) <= tol, "n=%zu at_end=%d got %.9g want %.9g (tol %.3g)",
                  n, at_end, (double)got, want, tol);
            guarded_free(&ga);
            guarded_free(&gb);
        }
    }
}

static void test_dot_heap(ai_dot_fn dot, ai_rng *rng)
{
    for (size_t k = 0; k < N_SIZES; ++k) {
        const size_t n = test_size(k);
        for (size_t oa = 0; oa < 8; oa += 3) {
            for (size_t ob = 0; ob < 8; ob += 5) {
                float *a = malloc((n + 24) * sizeof(float));
                float *b = malloc((n + 24) * sizeof(float));
                if (!a || !b) {
                    CHECK(0, "malloc failed");
                    return;
                }
                for (size_t i = 0; i < n + 24; ++i)           /* poison everything ... */
                    a[i] = b[i] = make_nan();
                fill_random(a + oa, n, rng);                  /* ... then open a window */
                fill_random(b + ob, n, rng);
                double as;
                const double want = dot_ref(a + oa, b + ob, n, &as);
                const float  got  = dot(a + oa, b + ob, n);   /* any stray read -> NaN */
                const double tol  = 1e-5 * as + 1e-7;
                CHECK(fabs((double)got - want) <= tol, "n=%zu oa=%zu ob=%zu got %.9g want %.9g",
                      n, oa, ob, (double)got, want);
                free(a);
                free(b);
            }
        }
    }
}

/* ============================== axpy ========================================== */

static int axpy_matches(size_t n, float alpha, const float *x, const float *y0, const float *y,
                        size_t *bad)
{
    for (size_t i = 0; i < n; ++i) {
        const double want = (double)y0[i] + (double)alpha * (double)x[i];
        const double tol  = 1e-6 * (fabs((double)y0[i]) + fabs((double)alpha * (double)x[i])) + 1e-30;
        if (!(fabs((double)y[i] - want) <= tol)) {
            *bad = i;
            return 0;
        }
    }
    return 1;
}

static void test_axpy_guard(ai_axpy_fn axpy, ai_rng *rng)
{
    for (size_t k = 0; k < N_SIZES; ++k) {
        const size_t n = test_size(k);
        for (int at_end = 0; at_end <= 1; ++at_end) {
            guarded gx, gy;
            float *x0 = malloc((n + 1) * sizeof(float));
            float *y0 = malloc((n + 1) * sizeof(float));
            if (guarded_alloc(&gx, n, at_end) || guarded_alloc(&gy, n, at_end) || !x0 || !y0) {
                CHECK(0, "allocation failed");
                return;
            }
            fill_random(gx.p, n, rng);
            fill_random(gy.p, n, rng);
            memcpy(x0, gx.p, n * sizeof(float));
            memcpy(y0, gy.p, n * sizeof(float));
            const float alpha = 3.0f * ai_rng_uniform(rng) - 1.5f;

            axpy(n, alpha, gx.p, gy.p);

            size_t bad = 0;
            CHECK(axpy_matches(n, alpha, x0, y0, gy.p, &bad),
                  "n=%zu at_end=%d y[%zu]=%.9g, y0=%.9g x=%.9g alpha=%.9g",
                  n, at_end, bad, (double)gy.p[bad], (double)y0[bad], (double)x0[bad], (double)alpha);
            CHECK(memcmp(x0, gx.p, n * sizeof(float)) == 0, "n=%zu: x was modified", n);
            free(x0);
            free(y0);
            guarded_free(&gx);
            guarded_free(&gy);
        }
    }
}

static void test_axpy_heap(ai_axpy_fn axpy, ai_rng *rng)
{
    for (size_t k = 0; k < N_SIZES; ++k) {
        const size_t n = test_size(k);
        for (size_t ox = 0; ox < 8; ox += 3) {
            for (size_t oy = 0; oy < 8; oy += 5) {
                const size_t total = n + 24;
                float *x   = malloc(total * sizeof(float));
                float *y   = malloc(total * sizeof(float));
                float *y0  = malloc(total * sizeof(float));
                if (!x || !y || !y0) {
                    CHECK(0, "malloc failed");
                    return;
                }
                fill_random(x, total, rng);
                fill_random(y, total, rng);
                memcpy(y0, y, total * sizeof(float));
                const float alpha = 3.0f * ai_rng_uniform(rng) - 1.5f;

                axpy(n, alpha, x + ox, y + oy);

                size_t bad = 0;
                CHECK(axpy_matches(n, alpha, x + ox, y0 + oy, y + oy, &bad),
                      "n=%zu ox=%zu oy=%zu y[%zu] wrong", n, ox, oy, bad);
                /* every byte outside the written window must be bit-identical */
                CHECK(memcmp(y, y0, oy * sizeof(float)) == 0,
                      "n=%zu oy=%zu: wrote BEFORE y[0]", n, oy);
                CHECK(memcmp(y + oy + n, y0 + oy + n, (total - oy - n) * sizeof(float)) == 0,
                      "n=%zu oy=%zu: wrote PAST y[n-1]", n, oy);

                axpy(n, 0.0f, x + ox, y + oy);                 /* alpha = 0 changes nothing */
                CHECK(memcmp(y + oy + n, y0 + oy + n, (total - oy - n) * sizeof(float)) == 0,
                      "n=%zu: alpha=0 wrote past the end", n);
                free(x);
                free(y);
                free(y0);
            }
        }
    }
}

/* ============================== ABI =========================================== */

/* asm/abi_check.asm: calls fn with canaries in every non-volatile register and
 * returns a bitmask of the ones that changed (0 = ABI honoured). */
extern unsigned abi_check(size_t fn, size_t a, size_t b, size_t c, size_t d, float alpha);
extern void     abi_selftest_clobber(void);

static void test_abi_harness(void)
{
    /* The harness must (1) stay silent on code that follows the rules ... */
    float v[64] = { 0 };
    const unsigned clean = abi_check((size_t)ai_dot_scalar, (size_t)v, (size_t)v, 64, 0, 0.0f);
    CHECK(clean == 0, "harness false alarm on gcc-compiled code: mask 0x%x", clean);
    /* ... and (2) notice when they are broken: RBX = bit 0, R12 = bit 4, XMM6 = bit 8. */
    const unsigned broken = abi_check((size_t)abi_selftest_clobber, 0, 0, 0, 0, 0.0f);
    CHECK(broken == 0x111u, "harness missed a deliberate violation: mask 0x%x (want 0x111)", broken);
}

static void test_abi_avx2(void)
{
    static float x[256], y[256];
    CHECK(abi_check((size_t)ai_cpu_has_avx2_fma, 0, 0, 0, 0, 0.0f) == 0,
          "ai_cpu_has_avx2_fma clobbered a non-volatile register (RBX?)");
    static const size_t ns[] = { 0, 1, 7, 8, 9, 31, 32, 33, 100, 255 };
    for (size_t k = 0; k < sizeof ns / sizeof *ns; ++k) {
        const size_t n = ns[k];
        const unsigned md = abi_check((size_t)ai_dot_avx2, (size_t)x, (size_t)y, n, 0, 0.0f);
        CHECK(md == 0, "ai_dot_avx2(n=%zu) clobbered non-volatile registers: mask 0x%x", n, md);
        const unsigned ma = abi_check((size_t)ai_axpy_avx2, n, 0, (size_t)x, (size_t)y, 0.5f);
        CHECK(ma == 0, "ai_axpy_avx2(n=%zu) clobbered non-volatile registers: mask 0x%x", n, ma);
    }
}

/* ============================== driver ======================================== */

#define RUN(label, call)                                                       \
    do {                                                                       \
        const long f0 = g_fails;                                               \
        printf("  %-52s", label);                                              \
        fflush(stdout);                                                        \
        call;                                                                  \
        printf("%s\n", g_fails == f0 ? "ok" : "\n  ... FAILED");               \
        fflush(stdout);                                                        \
    } while (0)

static void run_suite(const char *name, ai_dot_fn dot, ai_axpy_fn axpy)
{
    ai_rng rng;
    ai_rng_seed(&rng, 20260609);
    printf("[%s]\n", name);
    RUN("dot : exact integer data, n = 0..100",               test_dot_exact(dot));
    RUN("dot : guard pages, 82 sizes x both ends",            test_dot_guard(dot, &rng));
    RUN("dot : unaligned + NaN-poisoned neighbours",          test_dot_heap(dot, &rng));
    RUN("axpy: guard pages, 82 sizes x both ends",            test_axpy_guard(axpy, &rng));
    RUN("axpy: unaligned, untouched surroundings, alpha = 0", test_axpy_heap(axpy, &rng));
}

int main(void)
{
    /* A kernel that strays onto a guard page is SUPPOSED to crash this process
     * (exit code 0xC0000005 = access violation).  Keep Windows from putting up a
     * "has stopped working" dialog about it. */
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    const unsigned hw = ai_cpu_has_avx2_fma();
    __builtin_cpu_init();
    const unsigned gcc_says = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    printf("CPU feature detection: asm says %u, gcc's cpuid says %u\n", hw, gcc_says);
    CHECK(hw == gcc_says, "ai_cpu_has_avx2_fma() disagrees with __builtin_cpu_supports");

    printf("[ABI harness]\n");
    RUN("canary registers: silent on good code, loud on bad", test_abi_harness());

    run_suite("scalar C reference (validates the test harness itself)", ai_dot_scalar, ai_axpy_scalar);

    if (hw) {
        run_suite("AVX2+FMA assembly", ai_dot_avx2, ai_axpy_avx2);
        RUN("ABI: RBX RBP RDI RSI R12-R15 XMM6-15 preserved", test_abi_avx2());
    } else
        printf("[AVX2+FMA assembly] SKIPPED: this CPU/OS does not support AVX2+FMA\n");

    printf("\n%ld checks, %ld failed -> %s\n", g_checks, g_fails, g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
