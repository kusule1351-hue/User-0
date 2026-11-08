/* mlp.h - a small fully-connected neural network (multi-layer perceptron).
 *
 *   x -> [W0 b0] -> ReLU -> [W1 b1] -> ReLU -> ... -> [Wn bn] -> softmax -> loss
 *
 * Every weight matrix is stored ROW-MAJOR as (out x in).  That one decision is
 * what makes the whole network two kernels:
 *
 *   forward : z[j]   = dot (W_j, input) + b[j]          one dot  per neuron
 *   backward: dprev += delta[j] * W_j                   one axpy per neuron
 *             gW_j   += delta[j] * input                one axpy per neuron
 *   update  : params -= lr * grads                      one long axpy
 *
 * where W_j is row j, contiguous in memory.  See mlp.c.
 */
#ifndef TINYAI_MLP_H
#define TINYAI_MLP_H

#include <stddef.h>

#include "kernels.h"
#include "rng.h"

#define AI_MAX_LAYERS 8

typedef struct ai_mlp {
    const ai_backend *be;
    size_t n_layers;                    /* weight layers = hidden layers + output */
    size_t dims[AI_MAX_LAYERS + 1];     /* dims[0] = inputs ... dims[n_layers] = classes */
    size_t n_params;

    float *params;                      /* [W0 b0 W1 b1 ...] in one block         */
    float *grads;                       /* same layout; accumulated by loss_grad  */
    float *W[AI_MAX_LAYERS],  *b[AI_MAX_LAYERS];
    float *gW[AI_MAX_LAYERS], *gb[AI_MAX_LAYERS];

    float *z[AI_MAX_LAYERS];            /* pre-activation of layer l, dims[l+1]   */
    float *a[AI_MAX_LAYERS];            /* post-ReLU of layer l (hidden only)     */
    float *d[AI_MAX_LAYERS];            /* dLoss/dz of layer l                    */
    float *workspace;                   /* owns the z/a/d storage                 */
} ai_mlp;

/* dims[0..n_layers] gives the layer widths, e.g. {2, 64, 64, 3} with n_layers=3.
 * Weights get He-normal initialisation (right for ReLU), biases start at zero.
 * Returns 0 on success, -1 on bad arguments or out of memory. */
int ai_mlp_init(ai_mlp *m, const size_t *dims, size_t n_layers,
                const ai_backend *be, ai_rng *rng);
void ai_mlp_free(ai_mlp *m);

/* Forward pass; returns the logits (valid until the next call). */
const float *ai_mlp_forward(ai_mlp *m, const float *x);

/* Index of the largest logit. */
size_t ai_mlp_predict(ai_mlp *m, const float *x);

/* softmax + cross-entropy of one sample.  Writes the probabilities to `probs`
 * if it is non-NULL.  Returns -log(p[label]). */
float ai_softmax_xent(const float *logits, size_t k, size_t label, float *probs);

/* Forward + backward for one labelled sample.  ADDS its gradient to m->grads
 * (it does not overwrite), so several calls build up a mini-batch gradient.
 * Returns the sample's loss.  Parameters are not touched. */
float ai_mlp_loss_grad(ai_mlp *m, const float *x, size_t label);

/* params -= (lr / batch) * grads, then clears grads. */
void ai_mlp_sgd_step(ai_mlp *m, float lr, size_t batch);

#endif /* TINYAI_MLP_H */
