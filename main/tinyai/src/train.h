/* train.h - the training loop pieces shared by the CLI (main.c) and by the
 * foreign-language API (tinyai_api.c, used from Lisp): one SGD pass over a
 * dataset, evaluation, and the learning-rate schedule. */
#ifndef TINYAI_TRAIN_H
#define TINYAI_TRAIN_H

#include <stddef.h>

#include "data.h"
#include "mlp.h"
#include "rng.h"

/* One SGD pass over `ds`.  `order` holds ds->n sample indices (start it as
 * 0,1,2,...); it is reshuffled IN PLACE first (Fisher-Yates driven by `shuffle`),
 * so pass the same array every epoch.  Weights are updated every `batch` samples
 * (the last update of a pass may use fewer).  Returns the mean training loss
 * measured DURING the pass (weights are changing, so it lags the true loss). */
double ai_train_epoch(ai_mlp *m, const ai_dataset *ds, size_t *order, ai_rng *shuffle,
                      float lr, size_t batch);

/* Mean loss and accuracy (percent) of the current weights on `ds`; no learning. */
void ai_evaluate(ai_mlp *m, const ai_dataset *ds, double *loss, double *acc_percent);

/* Cosine decay from `lr` (epoch 1) towards lr * lr_final (last epoch):
 * big steps early to find the rough shape, small steps late so single noisy
 * samples stop knocking the weights around.  `epoch` counts from 1. */
float ai_cosine_lr(float lr, float lr_final, size_t epoch, size_t epochs);

#endif /* TINYAI_TRAIN_H */
