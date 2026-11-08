/* timer.h - high-resolution wall clock (Windows QueryPerformanceCounter). */
#ifndef TINYAI_TIMER_H
#define TINYAI_TIMER_H

#include <windows.h>

static inline double ai_now(void)
{
    static double inv_freq = 0.0;
    if (inv_freq == 0.0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        inv_freq = 1.0 / (double)f.QuadPart;
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * inv_freq;
}

#endif /* TINYAI_TIMER_H */
