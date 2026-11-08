/* tinyai_api.c - flat C interface for foreign callers (see tinyai_api.h). */
#include "tinyai_api.h"

#include <math.h>
#include <stdlib.h>

#include "data.h"
#include "kernels.h"
#include "mlp.h"
#include "rng.h"
#include "train.h"

struct tai_data {
    ai_dataset ds;
};

struct tai_net {
    ai_mlp  mlp;
    ai_rng  shuffle;
    size_t *order;          /* visiting order, reshuffled every epoch */
    size_t  order_n;
};

/* ---- kernels ------------------------------------------------------------------- */
static const ai_backend *best(void)
{
    static const ai_backend *cached;
    if (!cached)
        cached = ai_select_backend(0);
    return cached;
}

uint32_t tai_cpu_has_avx2_fma(void)                  { return ai_cpu_has_avx2_fma(); }
const char *tai_kernel_name(void)                    { return best()->name; }
float tai_dot(const float *a, const float *b, uint64_t n)
{
    return best()->dot(a, b, (size_t)n);
}
void tai_axpy(uint64_t n, float alpha, const float *x, float *y)
{
    best()->axpy((size_t)n, alpha, x, y);
}
float tai_dot_scalar(const float *a, const float *b, uint64_t n)
{
    return ai_dot_scalar(a, b, (size_t)n);
}
void tai_axpy_scalar(uint64_t n, float alpha, const float *x, float *y)
{
    ai_axpy_scalar((size_t)n, alpha, x, y);
}

/* ---- datasets ------------------------------------------------------------------ */
tai_data *tai_spirals_new(uint64_t per_class, uint64_t classes, float turns, float noise,
                          uint64_t seed)
{
    if (per_class == 0 || classes == 0 || per_class > (1u << 24) || classes > 64)
        return NULL;
    tai_data *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    ai_rng rng;
    ai_rng_seed(&rng, seed);
    if (ai_make_spirals(&d->ds, (size_t)per_class, (size_t)classes, turns, noise, &rng)) {
        free(d);
        return NULL;
    }
    return d;
}

void tai_data_free(tai_data *d)
{
    if (!d)
        return;
    ai_dataset_free(&d->ds);
    free(d);
}

uint64_t tai_data_size(const tai_data *d)    { return d ? d->ds.n : 0; }
uint64_t tai_data_dim(const tai_data *d)     { return d ? d->ds.dim : 0; }
uint64_t tai_data_classes(const tai_data *d) { return d ? d->ds.classes : 0; }
const float *tai_data_x(const tai_data *d)   { return d ? d->ds.x : NULL; }
const uint32_t *tai_data_y(const tai_data *d){ return d ? (const uint32_t *)d->ds.y : NULL; }

/* ---- networks ------------------------------------------------------------------ */
tai_net *tai_net_new(const uint64_t *dims, uint64_t n_layers, uint64_t seed, int32_t force_scalar)
{
    if (!dims || n_layers == 0 || n_layers > AI_MAX_LAYERS)
        return NULL;
    size_t d[AI_MAX_LAYERS + 1];
    for (uint64_t i = 0; i <= n_layers; ++i) {
        if (dims[i] == 0 || dims[i] > (1u << 24))
            return NULL;
        d[i] = (size_t)dims[i];
    }
    tai_net *n = calloc(1, sizeof *n);
    if (!n)
        return NULL;
    ai_rng init;
    ai_rng_seed(&init, seed);
    ai_rng_seed(&n->shuffle, seed + 1);
    if (ai_mlp_init(&n->mlp, d, (size_t)n_layers, ai_select_backend(force_scalar), &init)) {
        free(n);
        return NULL;
    }
    return n;
}

void tai_net_free(tai_net *n)
{
    if (!n)
        return;
    ai_mlp_free(&n->mlp);
    free(n->order);
    free(n);
}

uint64_t tai_net_n_params(const tai_net *n) { return n ? n->mlp.n_params : 0; }
uint64_t tai_net_inputs(const tai_net *n)   { return n ? n->mlp.dims[0] : 0; }
uint64_t tai_net_outputs(const tai_net *n)  { return n ? n->mlp.dims[n->mlp.n_layers] : 0; }
const char *tai_net_backend(const tai_net *n) { return n ? n->mlp.be->name : ""; }

/* A dataset fits a network when the feature counts agree and every label is a
 * valid output index (otherwise the loss would read past the logits). */
static int fits(const tai_net *n, const tai_data *d)
{
    return n && d && d->ds.dim == n->mlp.dims[0] && d->ds.classes <= n->mlp.dims[n->mlp.n_layers];
}

double tai_net_train_epoch(tai_net *n, const tai_data *d, float lr, uint64_t batch)
{
    if (!fits(n, d))
        return NAN;
    if (n->order_n != d->ds.n) {                         /* new dataset size: restart the order */
        size_t *o = realloc(n->order, d->ds.n * sizeof *o);
        if (!o)
            return NAN;
        n->order = o;
        n->order_n = d->ds.n;
        for (size_t i = 0; i < n->order_n; ++i)
            n->order[i] = i;
    }
    return ai_train_epoch(&n->mlp, &d->ds, n->order, &n->shuffle, lr, (size_t)batch);
}

void tai_net_evaluate(tai_net *n, const tai_data *d, double *loss, double *acc_percent)
{
    if (!loss || !acc_percent)
        return;
    if (!fits(n, d)) {
        *loss = *acc_percent = NAN;
        return;
    }
    ai_evaluate(&n->mlp, &d->ds, loss, acc_percent);
}

uint64_t tai_net_predict(tai_net *n, const float *x)
{
    return (n && x) ? ai_mlp_predict(&n->mlp, x) : 0;
}

const float *tai_net_logits(tai_net *n, const float *x)
{
    return (n && x) ? ai_mlp_forward(&n->mlp, x) : NULL;
}
