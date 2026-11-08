/* main.c - train a neural network on 3-class spirals and show what it learned.
 *
 *   tinyai [--epochs N] [--lr X] [--lr-final X] [--hidden H] [--batch B]
 *          [--seed S] [--per-class N] [--turns X] [--noise X] [--log-every N]
 *          [--scalar] [--no-plot]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "kernels.h"
#include "mlp.h"
#include "rng.h"
#include "timer.h"
#include "train.h"

typedef struct options {
    size_t    epochs, hidden, batch, per_class, log_every;
    float     lr, lr_final, noise, turns;
    uint64_t  seed;
    int       force_scalar, plot;
} options;

static void usage(const char *prog)
{
    printf("usage: %s [options]\n"
           "  --epochs N      training passes over the data     (default 300)\n"
           "  --lr X          SGD learning rate at the start     (default 0.02)\n"
           "  --lr-final X    fraction of --lr reached by the last epoch, via cosine\n"
           "                  decay; 1 = constant rate           (default 0.05)\n"
           "  --hidden H      width of both hidden layers        (default 64)\n"
           "  --batch B       samples per weight update          (default 1)\n"
           "  --per-class N   training points per spiral arm     (default 200)\n"
           "  --turns X       revolutions per spiral arm         (default 1.0)\n"
           "  --noise X       angular noise, radians (arms are 2.09 apart) (default 0.40)\n"
           "  --seed S        RNG seed                           (default 1234)\n"
           "  --log-every N   print a line every N epochs        (default 25)\n"
           "  --scalar        force the plain-C kernels (also: set AI_FORCE_SCALAR=1)\n"
           "  --no-plot       skip the ASCII decision-region picture\n",
           prog);
}

static int parse_args(int argc, char **argv, options *o)
{
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const int has_val = (i + 1 < argc);
        if (!strcmp(a, "--scalar"))        { o->force_scalar = 1; }
        else if (!strcmp(a, "--no-plot"))  { o->plot = 0; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 1; }
        else if (!strcmp(a, "--epochs")    && has_val) { o->epochs    = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--hidden")    && has_val) { o->hidden    = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--batch")     && has_val) { o->batch     = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--per-class") && has_val) { o->per_class = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--log-every") && has_val) { o->log_every = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--seed")      && has_val) { o->seed      = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--lr")        && has_val) { o->lr        = (float)strtod(argv[++i], NULL); }
        else if (!strcmp(a, "--lr-final")  && has_val) { o->lr_final  = (float)strtod(argv[++i], NULL); }
        else if (!strcmp(a, "--noise")     && has_val) { o->noise     = (float)strtod(argv[++i], NULL); }
        else if (!strcmp(a, "--turns")     && has_val) { o->turns     = (float)strtod(argv[++i], NULL); }
        else {
            fprintf(stderr, "unknown or incomplete option: %s\n\n", a);
            usage(argv[0]);
            return -1;
        }
    }
    if (o->epochs == 0 || o->hidden == 0 || o->batch == 0 || o->per_class == 0 ||
        o->log_every == 0 || !(o->lr > 0.0f)) {
        fprintf(stderr, "epochs, hidden, batch, per-class, log-every and lr must be positive\n");
        return -1;
    }
    if (!(o->lr_final > 0.0f && o->lr_final <= 1.0f)) {
        fprintf(stderr, "lr-final must be in (0, 1]\n");
        return -1;
    }
    return 0;
}

/* Colour the plane by the class the network predicts, then drop the test points
 * on top: a digit is the point's true class, '!' marks a misclassified one. */
static void plot_regions(ai_mlp *m, const ai_dataset *pts)
{
    enum { W = 66, H = 33 };
    const float lim = 1.15f;
    static const char region[] = " .:-=+*#";          /* one glyph per class (<= 8) */
    static char grid[H][W + 1];

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float p[2];
            p[0] = -lim + 2.0f * lim * ((float)c + 0.5f) / (float)W;
            p[1] =  lim - 2.0f * lim * ((float)r + 0.5f) / (float)H;
            grid[r][c] = region[ai_mlp_predict(m, p) % (sizeof region - 1)];
        }
        grid[r][W] = '\0';
    }
    for (size_t i = 0; i < pts->n; ++i) {
        const float *p = pts->x + i * 2;
        const int c = (int)((p[0] + lim) / (2.0f * lim) * (float)W);
        const int r = (int)((lim - p[1]) / (2.0f * lim) * (float)H);
        if (c < 0 || c >= W || r < 0 || r >= H)
            continue;
        grid[r][c] = (ai_mlp_predict(m, p) == pts->y[i]) ? (char)('0' + pts->y[i] % 10) : '!';
    }

    printf("\nWhat the network learned (shading = predicted class, digits = test points):\n");
    printf("+"); for (int c = 0; c < W; ++c) putchar('-'); printf("+\n");
    for (int r = 0; r < H; ++r)
        printf("|%s|\n", grid[r]);
    printf("+"); for (int c = 0; c < W; ++c) putchar('-'); printf("+\n");
    printf("class 0 = blank   class 1 = '.'   class 2 = ':'     '!' = misclassified test point\n");
}

int main(int argc, char **argv)
{
    options o = { .epochs = 300, .hidden = 64, .batch = 1, .per_class = 200, .log_every = 25,
                  .lr = 0.02f, .lr_final = 0.05f, .noise = 0.40f, .turns = 1.00f, .seed = 1234,
                  .force_scalar = 0, .plot = 1 };
    const int pr = parse_args(argc, argv, &o);
    if (pr != 0)
        return pr > 0 ? 0 : 2;

    const ai_backend *be = ai_select_backend(o.force_scalar);
    const size_t classes = 3;

    ai_rng data_rng, test_rng, init_rng, shuffle_rng;
    ai_rng_seed(&data_rng,    o.seed);
    ai_rng_seed(&test_rng,    o.seed + 1);
    ai_rng_seed(&init_rng,    o.seed + 2);
    ai_rng_seed(&shuffle_rng, o.seed + 3);

    ai_dataset train, test;
    if (ai_make_spirals(&train, o.per_class, classes, o.turns, o.noise, &data_rng) ||
        ai_make_spirals(&test,  o.per_class, classes, o.turns, o.noise, &test_rng)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    const size_t dims[4] = { 2, o.hidden, o.hidden, classes };
    ai_mlp net;
    if (ai_mlp_init(&net, dims, 3, be, &init_rng)) {
        fprintf(stderr, "could not create the network\n");
        return 1;
    }
    size_t *order = malloc(train.n * sizeof *order);
    if (!order) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (size_t i = 0; i < train.n; ++i)
        order[i] = i;

    printf("tinyai - a neural network with C + x86-64 assembly kernels\n");
    printf("backend : %s%s\n", be->name,
           (be == &ai_backend_scalar && ai_cpu_has_avx2_fma()) ? "   (AVX2 available, not used)" : "");
    printf("data    : %zu-class spirals, %zu train / %zu test points, %.2f turns, noise %.2f\n",
           classes, train.n, test.n, (double)o.turns, (double)o.noise);
    printf("model   : 2 -> %zu -> %zu -> %zu  (%zu parameters), ReLU, softmax cross-entropy\n",
           o.hidden, o.hidden, classes, net.n_params);
    printf("training: SGD, lr %g -> %g (cosine), batch %zu, %zu epochs, seed %llu\n\n",
           (double)o.lr, (double)(o.lr * o.lr_final), o.batch, o.epochs,
           (unsigned long long)o.seed);
    printf(" epoch        lr       loss    train%%     test%%\n");

    double train_seconds = 0.0;
    for (size_t epoch = 1; epoch <= o.epochs; ++epoch) {
        const float lr = ai_cosine_lr(o.lr, o.lr_final, epoch, o.epochs);

        const double t0 = ai_now();
        ai_train_epoch(&net, &train, order, &shuffle_rng, lr, o.batch);
        train_seconds += ai_now() - t0;

        if (epoch == 1 || epoch % o.log_every == 0 || epoch == o.epochs) {
            double tl, ta, el, ea;
            ai_evaluate(&net, &train, &tl, &ta);
            ai_evaluate(&net, &test,  &el, &ea);
            printf("%6zu  %8.5f  %9.5f  %7.2f%%  %7.2f%%\n", epoch, (double)lr, tl, ta, ea);
        }
    }

    double tl, ta, el, ea;
    ai_evaluate(&net, &train, &tl, &ta);
    ai_evaluate(&net, &test,  &el, &ea);
    printf("\nfinal   : train %.2f%%  test %.2f%%  (test loss %.4f)\n", ta, ea, el);
    printf("speed   : %.3f s of pure training, %.0f samples/s  [%s]\n", train_seconds,
           (double)(o.epochs * train.n) / train_seconds, be->name);

    if (o.plot)
        plot_regions(&net, &test);

    free(order);
    ai_mlp_free(&net);
    ai_dataset_free(&train);
    ai_dataset_free(&test);
    return 0;
}
