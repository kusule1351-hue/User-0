/* data.c - synthetic datasets. */
#include "data.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int ai_make_spirals(ai_dataset *ds, size_t per_class, size_t classes,
                    float turns, float noise, ai_rng *rng)
{
    if (!ds || !rng || per_class == 0 || classes == 0)
        return -1;
    memset(ds, 0, sizeof *ds);

    const size_t n = per_class * classes;
    ds->n = n;
    ds->dim = 2;
    ds->classes = classes;
    ds->x = malloc(n * 2 * sizeof(float));
    ds->y = malloc(n * sizeof(unsigned));
    if (!ds->x || !ds->y) {
        ai_dataset_free(ds);
        return -1;
    }

    const float two_pi = 6.28318530717958647692f;
    size_t idx = 0;
    for (size_t k = 0; k < classes; ++k) {
        for (size_t i = 0; i < per_class; ++i, ++idx) {
            const float frac = ai_rng_uniform(rng);          /* position along the arm */
            const float r = 0.05f + 0.95f * frac;            /* radius grows outward   */
            const float t = (float)k * (two_pi / (float)classes)   /* arm's start angle */
                          + two_pi * turns * frac            /* winding                */
                          + noise * ai_rng_gauss(rng);       /* jitter                 */
            ds->x[2 * idx + 0] = r * sinf(t);
            ds->x[2 * idx + 1] = r * cosf(t);
            ds->y[idx] = (unsigned)k;
        }
    }
    return 0;
}

void ai_dataset_free(ai_dataset *ds)
{
    if (!ds)
        return;
    free(ds->x);
    free(ds->y);
    memset(ds, 0, sizeof *ds);
}
