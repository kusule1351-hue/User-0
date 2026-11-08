/* mlp.c - forward pass, back-propagation and SGD, built on the dot/axpy kernels. */
#include "mlp.h"

#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 64-byte (cache-line) aligned, zeroed memory.  The kernels work on any
 * alignment, but a 32-byte vector access that straddles a cache line costs ~40%
 * throughput, and rows of a weight matrix are exactly what they stream through.
 * (_aligned_malloc is the UCRT call; C11 aligned_alloc is not supported there.) */
static float *alloc64(size_t n_floats)
{
    float *p = _aligned_malloc(n_floats * sizeof(float), 64);
    if (p)
        memset(p, 0, n_floats * sizeof(float));
    return p;
}

int ai_mlp_init(ai_mlp *m, const size_t *dims, size_t n_layers,
                const ai_backend *be, ai_rng *rng)
{
    if (!m || !dims || !be || !rng || n_layers == 0 || n_layers > AI_MAX_LAYERS)
        return -1;
    memset(m, 0, sizeof *m);
    m->be = be;
    m->n_layers = n_layers;

    size_t n_params = 0, n_act = 0;
    for (size_t l = 0; l <= n_layers; ++l) {
        if (dims[l] == 0)
            return -1;
        m->dims[l] = dims[l];
    }
    for (size_t l = 0; l < n_layers; ++l) {
        n_params += dims[l + 1] * dims[l] + dims[l + 1];
        n_act    += dims[l + 1];
    }
    m->n_params  = n_params;
    m->params    = alloc64(n_params);
    m->grads     = alloc64(n_params);
    m->workspace = alloc64(3 * n_act);
    if (!m->params || !m->grads || !m->workspace) {
        ai_mlp_free(m);
        return -1;
    }

    size_t off = 0, act = 0;
    for (size_t l = 0; l < n_layers; ++l) {
        const size_t ni = dims[l], no = dims[l + 1];
        m->W[l]  = m->params + off;
        m->gW[l] = m->grads  + off;
        off += no * ni;
        m->b[l]  = m->params + off;
        m->gb[l] = m->grads  + off;
        off += no;

        m->z[l] = m->workspace + act;
        m->a[l] = m->workspace + act + no;
        m->d[l] = m->workspace + act + 2 * no;
        act += 3 * no;

        /* He initialisation: N(0, 2/fan_in) keeps ReLU activations at a stable
         * scale from layer to layer, so training can start from step one. */
        const float std = sqrtf(2.0f / (float)ni);
        for (size_t i = 0; i < no * ni; ++i)
            m->W[l][i] = std * ai_rng_gauss(rng);
    }
    return 0;
}

void ai_mlp_free(ai_mlp *m)
{
    if (!m)
        return;
    _aligned_free(m->params);
    _aligned_free(m->grads);
    _aligned_free(m->workspace);
    memset(m, 0, sizeof *m);
}

const float *ai_mlp_forward(ai_mlp *m, const float *x)
{
    const ai_backend *be = m->be;
    const float *in = x;

    for (size_t l = 0; l < m->n_layers; ++l) {
        const size_t ni = m->dims[l], no = m->dims[l + 1];
        const float *W = m->W[l];
        float *z = m->z[l];

        for (size_t j = 0; j < no; ++j)                      /* the hot loop */
            z[j] = be->dot(W + j * ni, in, ni) + m->b[l][j];

        if (l + 1 < m->n_layers) {                           /* hidden layer: ReLU */
            float *a = m->a[l];
            for (size_t j = 0; j < no; ++j)
                a[j] = z[j] > 0.0f ? z[j] : 0.0f;
            in = a;
        }
    }
    return m->z[m->n_layers - 1];
}

size_t ai_mlp_predict(ai_mlp *m, const float *x)
{
    const float *logits = ai_mlp_forward(m, x);
    const size_t k = m->dims[m->n_layers];
    size_t best = 0;
    for (size_t i = 1; i < k; ++i)
        if (logits[i] > logits[best])
            best = i;
    return best;
}

float ai_softmax_xent(const float *logits, size_t k, size_t label, float *probs)
{
    float mx = logits[0];
    for (size_t i = 1; i < k; ++i)
        if (logits[i] > mx)
            mx = logits[i];

    /* Subtracting the max keeps expf() in range: without it logits of ~90
     * overflow to inf and the whole loss becomes NaN. */
    float sum = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        float e = expf(logits[i] - mx);
        if (probs)
            probs[i] = e;
        sum += e;
    }
    if (probs) {
        const float inv = 1.0f / sum;
        for (size_t i = 0; i < k; ++i)
            probs[i] *= inv;
    }
    return logf(sum) - (logits[label] - mx);                 /* -log p[label] */
}

float ai_mlp_loss_grad(ai_mlp *m, const float *x, size_t label)
{
    const ai_backend *be = m->be;
    const size_t L = m->n_layers;
    const size_t K = m->dims[L];

    /* ---- forward, then dLoss/dlogits = softmax(logits) - onehot(label) ---- */
    const float *logits = ai_mlp_forward(m, x);
    float *delta = m->d[L - 1];
    const float loss = ai_softmax_xent(logits, K, label, delta);
    delta[label] -= 1.0f;

    /* ---- backward ---------------------------------------------------------- */
    for (size_t l = L; l-- > 0; ) {
        const size_t ni = m->dims[l], no = m->dims[l + 1];
        const float *in = (l == 0) ? x : m->a[l - 1];
        const float *dl = m->d[l];

        /* Gradient w.r.t. this layer's input:  dprev = W^T * delta
         *   = sum_j delta[j] * W_j      (an axpy over each contiguous row)
         * then mask it with the ReLU derivative of the layer below.  The result
         * becomes `delta` of layer l-1.  (Layer 0's input is the data: no need.) */
        if (l > 0) {
            float *dprev = m->d[l - 1];
            memset(dprev, 0, ni * sizeof(float));
            for (size_t j = 0; j < no; ++j)
                be->axpy(ni, dl[j], m->W[l] + j * ni, dprev);
            const float *zp = m->z[l - 1];
            for (size_t i = 0; i < ni; ++i)
                if (!(zp[i] > 0.0f))
                    dprev[i] = 0.0f;
        }

        /* Gradient w.r.t. this layer's parameters:
         *   gW_j += delta[j] * input        gb[j] += delta[j]
         * Accumulated, not stored, so a mini-batch is just several calls. */
        for (size_t j = 0; j < no; ++j) {
            be->axpy(ni, dl[j], in, m->gW[l] + j * ni);
            m->gb[l][j] += dl[j];
        }
    }
    return loss;
}

void ai_mlp_sgd_step(ai_mlp *m, float lr, size_t batch)
{
    if (batch == 0)
        batch = 1;
    /* The entire parameter block is contiguous, so the update is ONE axpy. */
    m->be->axpy(m->n_params, -lr / (float)batch, m->grads, m->params);
    memset(m->grads, 0, m->n_params * sizeof(float));
}
