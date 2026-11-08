/* rng.h - tiny deterministic PRNG (xorshift64*), uniform and Gaussian samples.
 *
 * Own generator on purpose: rand() on the UCRT only yields 15 bits (RAND_MAX is
 * 32767) and differs between platforms; this one is identical everywhere, so a
 * given seed always gives the same data, the same initial weights, same shuffle. */
#ifndef TINYAI_RNG_H
#define TINYAI_RNG_H

#include <math.h>
#include <stdint.h>

typedef struct ai_rng {
    uint64_t s;
    float    spare;        /* second Box-Muller sample, saved for the next call */
    int      have_spare;
} ai_rng;

static inline uint64_t ai_splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void ai_rng_seed(ai_rng *r, uint64_t seed)
{
    uint64_t sm = seed;
    r->s = ai_splitmix64(&sm);          /* spreads low-entropy seeds like 1, 2, 3 */
    if (r->s == 0)
        r->s = 0x9E3779B97F4A7C15ull;   /* xorshift must never be all-zero */
    r->spare = 0.0f;
    r->have_spare = 0;
}

static inline uint64_t ai_rng_u64(ai_rng *r)
{
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* uniform in [0, 1) with 24 random bits (exactly representable as a float) */
static inline float ai_rng_uniform(ai_rng *r)
{
    return (float)(ai_rng_u64(r) >> 40) * (1.0f / 16777216.0f);
}

/* uniform integer in [0, n) */
static inline uint32_t ai_rng_below(ai_rng *r, uint32_t n)
{
    return (uint32_t)(((ai_rng_u64(r) >> 32) * (uint64_t)n) >> 32);
}

/* standard normal sample (Box-Muller) */
static inline float ai_rng_gauss(ai_rng *r)
{
    if (r->have_spare) {
        r->have_spare = 0;
        return r->spare;
    }
    float u1 = 1.0f - ai_rng_uniform(r);        /* (0, 1]: log() stays finite */
    float u2 = ai_rng_uniform(r);
    float m  = sqrtf(-2.0f * logf(u1));
    float t  = 6.28318530717958647692f * u2;
    r->spare = m * sinf(t);
    r->have_spare = 1;
    return m * cosf(t);
}

#endif /* TINYAI_RNG_H */
