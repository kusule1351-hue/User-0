/* data.h - labelled datasets. */
#ifndef TINYAI_DATA_H
#define TINYAI_DATA_H

#include <stddef.h>

#include "rng.h"

typedef struct ai_dataset {
    size_t    n;          /* number of samples            */
    size_t    dim;        /* features per sample          */
    size_t    classes;
    float    *x;          /* n * dim, row-major           */
    unsigned *y;          /* n class labels               */
} ai_dataset;

/* The classic "spiral" benchmark: `classes` interleaved arms winding out from
 * the origin.  No straight line (or any single linear layer) can separate them,
 * so the network has to learn a genuinely non-linear boundary.
 * `turns` is how many full revolutions each arm makes (more = harder),
 * `noise` the std-dev (radians) of the angular jitter.  Returns 0 / -1. */
int  ai_make_spirals(ai_dataset *ds, size_t per_class, size_t classes,
                     float turns, float noise, ai_rng *rng);
void ai_dataset_free(ai_dataset *ds);

#endif /* TINYAI_DATA_H */
