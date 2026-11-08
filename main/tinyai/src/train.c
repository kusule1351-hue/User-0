/* train.c - epoch loop, evaluation and learning-rate schedule. */
#include "train.h"

#include <math.h>

double ai_train_epoch(ai_mlp *m, const ai_dataset *ds, size_t *order, ai_rng *shuffle,
                      float lr, size_t batch)
{
    if (batch == 0)
        batch = 1;
    if (ds->n == 0)
        return 0.0;

    for (size_t i = ds->n; i > 1; --i) {                       /* Fisher-Yates shuffle */
        const size_t j = ai_rng_below(shuffle, (uint32_t)i);
        const size_t t = order[i - 1];
        order[i - 1] = order[j];
        order[j] = t;
    }

    double total = 0.0;
    size_t in_batch = 0;
    for (size_t s = 0; s < ds->n; ++s) {
        const size_t i = order[s];
        total += (double)ai_mlp_loss_grad(m, ds->x + i * ds->dim, ds->y[i]);
        if (++in_batch == batch) {
            ai_mlp_sgd_step(m, lr, in_batch);
            in_batch = 0;
        }
    }
    if (in_batch)
        ai_mlp_sgd_step(m, lr, in_batch);
    return total / (double)ds->n;
}

void ai_evaluate(ai_mlp *m, const ai_dataset *ds, double *loss, double *acc_percent)
{
    double total = 0.0;
    size_t correct = 0;
    for (size_t i = 0; i < ds->n; ++i) {
        const float *logits = ai_mlp_forward(m, ds->x + i * ds->dim);
        total += ai_softmax_xent(logits, ds->classes, ds->y[i], NULL);
        size_t best = 0;
        for (size_t c = 1; c < ds->classes; ++c)
            if (logits[c] > logits[best])
                best = c;
        correct += (best == ds->y[i]);
    }
    const double n = ds->n ? (double)ds->n : 1.0;
    *loss        = total / n;
    *acc_percent = 100.0 * (double)correct / n;
}

float ai_cosine_lr(float lr, float lr_final, size_t epoch, size_t epochs)
{
    const double progress = (double)(epoch - 1) / (double)epochs;       /* 0 .. <1 */
    return lr * (lr_final + (1.0f - lr_final) *
                 0.5f * (1.0f + (float)cos(3.14159265358979323846 * progress)));
}
