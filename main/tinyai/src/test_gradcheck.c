/* test_gradcheck.c - is back-propagation mathematically correct?
 *
 * The kernel tests prove that dot/axpy are right.  They cannot prove that the
 * NETWORK uses them correctly: a wrong transpose, a missing ReLU mask, a flipped
 * sign or a bias gradient that is never accumulated would all pass them.
 *
 * This test compares the analytic gradient from ai_mlp_loss_grad() against
 * central finite differences,   dL/dp ~ (L(p + e) - L(p - e)) / 2e,   of an
 * INDEPENDENT forward pass written below in double precision.  With e = 1e-6 the
 * finite difference is accurate to ~1e-9, so what remains is the float32
 * rounding of the code under test.  Every parameter (weights AND biases of all
 * three layers) is checked, over a mini-batch of three samples, on each backend.
 *
 * Also checked: gradients ACCUMULATE across calls (the mini-batch contract),
 * ai_mlp_sgd_step() applies p -= lr/batch * g and clears g, and the scalar and
 * AVX2 backends produce the same gradients.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "mlp.h"
#include "rng.h"

enum { NL = 3, NS = 3, MAXW = 16, MAXP = 256 };
static const size_t DIMS[NL + 1] = { 5, 7, 6, 4 };      /* 118 parameters */

static long g_checks, g_fails;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond) && ++g_fails <= 20) {                                      \
            printf("  FAIL (line %d): ", __LINE__);                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

/* Forward pass + softmax cross-entropy in double precision.  `p` has exactly the
 * layout of ai_mlp.params: [W0 b0 W1 b1 W2 b2], each W row-major (out x in).
 * `pattern` accumulates which ReLUs fired, so the caller can detect a finite
 * difference that straddles a ReLU kink (where the derivative does not exist). */
static double ref_loss(const double *p, const double *x, size_t label, unsigned long long *pattern)
{
    double cur[MAXW], nxt[MAXW];
    size_t off = 0;
    for (size_t i = 0; i < DIMS[0]; ++i)
        cur[i] = x[i];
    for (size_t l = 0; l < NL; ++l) {
        const size_t ni = DIMS[l], no = DIMS[l + 1];
        const double *W = p + off;
        const double *b = W + no * ni;
        off += no * ni + no;
        for (size_t j = 0; j < no; ++j) {
            double s = b[j];
            for (size_t i = 0; i < ni; ++i)
                s += W[j * ni + i] * cur[i];
            if (l + 1 < NL) {
                *pattern = *pattern * 1000003ull + (s > 0.0 ? 1u : 2u);
                nxt[j] = s > 0.0 ? s : 0.0;
            } else {
                nxt[j] = s;
            }
        }
        memcpy(cur, nxt, no * sizeof(double));
    }
    const size_t k = DIMS[NL];
    double mx = cur[0];
    for (size_t i = 1; i < k; ++i)
        if (cur[i] > mx)
            mx = cur[i];
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i)
        sum += exp(cur[i] - mx);
    return log(sum) + mx - cur[label];
}

typedef struct result {
    double grads[MAXP];
    size_t n;
} result;

static void run_backend(const ai_backend *be, result *out)
{
    printf("[%s]\n", be->name);

    ai_rng rng;
    ai_rng_seed(&rng, 42);
    ai_mlp m;
    if (ai_mlp_init(&m, DIMS, NL, be, &rng)) {
        CHECK(0, "ai_mlp_init failed");
        return;
    }
    for (size_t l = 0; l < NL; ++l)                       /* non-zero biases, so their */
        for (size_t j = 0; j < DIMS[l + 1]; ++j)          /* gradients are exercised   */
            m.b[l][j] = 0.3f * ai_rng_gauss(&rng);

    float  xf[NS][MAXW];
    double xd[NS][MAXW];
    size_t lab[NS];
    for (size_t s = 0; s < NS; ++s) {
        for (size_t i = 0; i < DIMS[0]; ++i) {
            xf[s][i] = ai_rng_gauss(&rng);
            xd[s][i] = (double)xf[s][i];
        }
        lab[s] = ai_rng_below(&rng, (uint32_t)DIMS[NL]);
    }

    /* ---- analytic: one mini-batch of NS samples ---------------------------- */
    double loss_f = 0.0;
    for (size_t s = 0; s < NS; ++s)
        loss_f += (double)ai_mlp_loss_grad(&m, xf[s], lab[s]);

    double pd[MAXP];
    const size_t np = m.n_params;
    CHECK(np <= MAXP, "test buffer too small");
    for (size_t p = 0; p < np; ++p)
        pd[p] = (double)m.params[p];                      /* the exact float values */

    unsigned long long pat = 0;
    double loss_d = 0.0;
    for (size_t s = 0; s < NS; ++s)
        loss_d += ref_loss(pd, xd[s], lab[s], &pat);
    CHECK(fabs(loss_f - loss_d) <= 1e-4 * fmax(1.0, fabs(loss_d)),
          "loss: float %.9g vs double %.9g", loss_f, loss_d);

    /* ---- numeric: central differences on every parameter -------------------- */
    const double eps = 1e-6;
    size_t skipped = 0;
    double max_rel = 0.0;
    for (size_t p = 0; p < np; ++p) {
        const double orig = pd[p];
        unsigned long long pp = 0, pm = 0;
        double lp = 0.0, lm = 0.0;
        pd[p] = orig + eps;
        for (size_t s = 0; s < NS; ++s)
            lp += ref_loss(pd, xd[s], lab[s], &pp);
        pd[p] = orig - eps;
        for (size_t s = 0; s < NS; ++s)
            lm += ref_loss(pd, xd[s], lab[s], &pm);
        pd[p] = orig;

        out->grads[p] = (double)m.grads[p];
        if (pp != pm) {                                   /* ReLU kink inside +-eps */
            ++skipped;
            continue;
        }
        const double num   = (lp - lm) / (2.0 * eps);
        const double ana   = (double)m.grads[p];
        const double scale = fmax(fabs(ana), fabs(num));
        CHECK(fabs(ana - num) <= 3e-3 * scale + 1e-5,
              "param %zu: analytic %.9g vs numeric %.9g", p, ana, num);
        if (scale > 1e-3 && fabs(ana - num) / scale > max_rel)
            max_rel = fabs(ana - num) / scale;
    }
    out->n = np;
    CHECK(skipped <= np / 20, "%zu of %zu parameters skipped at a ReLU kink", skipped, np);
    printf("  gradient vs finite differences: %zu params, %zu skipped (ReLU kink), "
           "max relative error %.2e\n", np, skipped, max_rel);

    /* ---- gradients accumulate: per-sample gradients must sum to the batch one */
    double ana_batch[MAXP], sum_single[MAXP];
    for (size_t p = 0; p < np; ++p) {
        ana_batch[p] = (double)m.grads[p];
        sum_single[p] = 0.0;
    }
    for (size_t s = 0; s < NS; ++s) {
        memset(m.grads, 0, np * sizeof(float));
        (void)ai_mlp_loss_grad(&m, xf[s], lab[s]);
        for (size_t p = 0; p < np; ++p)
            sum_single[p] += (double)m.grads[p];
    }
    int acc_ok = 1;
    for (size_t p = 0; p < np; ++p)
        if (fabs(sum_single[p] - ana_batch[p]) > 1e-5 * fabs(ana_batch[p]) + 1e-6)
            acc_ok = 0;
    CHECK(acc_ok, "sum of per-sample gradients != mini-batch gradient");
    printf("  gradients accumulate across calls: %s\n", acc_ok ? "ok" : "FAILED");

    /* ---- SGD step: p -= lr/batch * g, then g = 0 ------------------------------ */
    for (size_t s = 0; s < NS; ++s)
        (void)ai_mlp_loss_grad(&m, xf[s], lab[s]);        /* grads now hold the batch gradient */
    double before[MAXP], g[MAXP];
    for (size_t p = 0; p < np; ++p) {
        before[p] = (double)m.params[p];
        g[p] = (double)m.grads[p] ;
    }
    /* grads were zeroed by memset above before the last loop, so they hold exactly one batch */
    ai_mlp_sgd_step(&m, 0.1f, NS);
    int step_ok = 1, zero_ok = 1;
    for (size_t p = 0; p < np; ++p) {
        const double want = before[p] - 0.1 / (double)NS * g[p];
        if (fabs((double)m.params[p] - want) > 1e-6 * (fabs(want) + 1.0))
            step_ok = 0;
        if (m.grads[p] != 0.0f)
            zero_ok = 0;
    }
    CHECK(step_ok, "ai_mlp_sgd_step: parameters are not p - lr/batch * g");
    CHECK(zero_ok, "ai_mlp_sgd_step: gradients were not cleared");
    printf("  sgd step (p -= lr/batch*g) and gradient reset: %s\n",
           (step_ok && zero_ok) ? "ok" : "FAILED");

    ai_mlp_free(&m);
}

int main(void)
{
    static result r_scalar, r_avx2;

    run_backend(&ai_backend_scalar, &r_scalar);

    if (ai_cpu_has_avx2_fma()) {
        run_backend(&ai_backend_avx2, &r_avx2);
        int same = 1;
        double worst = 0.0;
        for (size_t p = 0; p < r_scalar.n; ++p) {
            const double d = fabs(r_scalar.grads[p] - r_avx2.grads[p]);
            const double tol = 1e-4 * fabs(r_scalar.grads[p]) + 1e-6;
            if (d > tol)
                same = 0;
            if (d / tol > worst)
                worst = d / tol;
        }
        CHECK(same, "scalar and AVX2 backends disagree on the gradients");
        printf("[scalar vs AVX2] gradients agree within tolerance: %s (worst = %.2f of the allowance)\n",
               same ? "ok" : "FAILED", worst);
    } else {
        printf("[AVX2+FMA] SKIPPED: this CPU/OS does not support AVX2+FMA\n");
    }

    printf("\n%ld checks, %ld failed -> %s\n", g_checks, g_fails, g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
