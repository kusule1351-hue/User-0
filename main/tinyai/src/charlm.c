/* charlm.c - a character-level language model on top of the tinyai MLP.
 *
 *   previous CTX characters (one-hot, CTX*V inputs)
 *     -> ReLU(H) -> ReLU(H) -> softmax over the V-symbol alphabet
 *
 * Bengio-style: the first layer's weight columns are, in effect, learned
 * character embeddings per position.  Same C + AVX2 kernels as the spiral demo.
 *
 *   charlm train [--corpus F] [--out F] [--hidden H] [--samples N] [--lr X]
 *                [--lr-final X] [--batch B] [--seed S]
 *   charlm gen   [--model F] [--prompt TEXT] [--chars N] [--temp T] [--topk K] [--seed S]
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "mlp.h"
#include "rng.h"
#include "timer.h"
#include "train.h"

#define CTX 16
static const char ALPHABET[] = " .,;:?!'-abcdefghijklmnopqrstuvwxyz";
#define V ((int)(sizeof ALPHABET - 1))
#define VAL_BLOCK 1000      /* every 20th block of 1000 characters is held out */
#define VAL_EVERY 20

static int sym_of(int c)
{
    const char *p = strchr(ALPHABET, c);
    return (c && p) ? (int)(p - ALPHABET) : -1;
}

static void encode_context(const unsigned char *text, size_t end, float *x)
{
    memset(x, 0, sizeof(float) * CTX * V);
    for (int j = 0; j < CTX; ++j)
        x[j * V + text[end - CTX + (size_t)j]] = 1.0f;
}

static int is_val(size_t pos) { return (pos / VAL_BLOCK) % VAL_EVERY == 0; }

/* ---- model file: "CLM1", CTX, V, n_layers, dims[n_layers+1], params ---------- */
static int save_model(const char *path, const ai_mlp *m)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t hdr[3] = { CTX, (uint32_t)V, (uint32_t)m->n_layers };
    fwrite("CLM1", 1, 4, f);
    fwrite(hdr, sizeof hdr, 1, f);
    for (size_t i = 0; i <= m->n_layers; ++i) { uint32_t d = (uint32_t)m->dims[i]; fwrite(&d, 4, 1, f); }
    fwrite(m->params, sizeof(float), m->n_params, f);
    return fclose(f);
}

static int load_model(const char *path, ai_mlp *m, const ai_backend *be)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4]; uint32_t hdr[3]; size_t dims[AI_MAX_LAYERS + 1];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CLM1", 4) || fread(hdr, sizeof hdr, 1, f) != 1 ||
        hdr[0] != CTX || hdr[1] != (uint32_t)V || hdr[2] < 1 || hdr[2] > AI_MAX_LAYERS) { fclose(f); return -2; }
    for (uint32_t i = 0; i <= hdr[2]; ++i) { uint32_t d; if (fread(&d, 4, 1, f) != 1) { fclose(f); return -2; } dims[i] = d; }
    ai_rng r; ai_rng_seed(&r, 1);
    if (ai_mlp_init(m, dims, hdr[2], be, &r)) { fclose(f); return -3; }
    if (fread(m->params, sizeof(float), m->n_params, f) != m->n_params) { fclose(f); return -2; }
    fclose(f);
    return 0;
}

/* ---- corpus -------------------------------------------------------------------- */
static unsigned char *load_corpus(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *t = malloc((size_t)sz + 1);
    if (!t || fread(t, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(t); return NULL; }
    fclose(f);
    size_t k = 0;
    for (long i = 0; i < sz; ++i) { int s = sym_of(t[i]); if (s >= 0) t[k++] = (unsigned char)s; }
    *n = k;
    return t;
}

/* mean cross-entropy (nats/char) and next-char accuracy on up to `max` held-out samples */
static void eval_val(ai_mlp *m, const unsigned char *text, const size_t *val, size_t nval, size_t max,
                     float *x, double *loss, double *acc)
{
    size_t step = nval > max ? nval / max : 1, cnt = 0, hit = 0;
    double sum = 0.0;
    for (size_t i = 0; i < nval; i += step) {
        size_t p = val[i];
        encode_context(text, p, x);
        const float *lg = ai_mlp_forward(m, x);
        sum += ai_softmax_xent(lg, (size_t)V, text[p], NULL);
        hit += ai_mlp_predict(m, x) == text[p];
        ++cnt;
    }
    *loss = sum / (double)cnt;
    *acc = 100.0 * (double)hit / (double)cnt;
}

static int cmd_train(const char *corpus, const char *out, size_t hidden, size_t samples, float lr,
                     float lr_final, size_t batch, uint64_t seed)
{
    size_t n;
    unsigned char *text = load_corpus(corpus, &n);
    if (!text) return 1;

    size_t *trn = malloc(n * sizeof *trn), *val = malloc(n * sizeof *val), nt = 0, nv = 0;
    for (size_t p = CTX; p < n; ++p) { if (is_val(p)) val[nv++] = p; else trn[nt++] = p; }

    const ai_backend *be = ai_select_backend(0);
    ai_rng init_rng, shuf; ai_rng_seed(&init_rng, seed); ai_rng_seed(&shuf, seed + 1);
    const size_t dims[4] = { CTX * V, hidden, hidden, V };
    ai_mlp m;
    if (ai_mlp_init(&m, dims, 3, be, &init_rng)) { fprintf(stderr, "init failed\n"); return 1; }
    float *x = malloc(sizeof(float) * CTX * V);

    printf("charlm - character-level language model on the tinyai C + assembly MLP\n");
    printf("backend  : %s\n", be->name);
    printf("corpus   : %zu chars, %d symbols;  %zu train / %zu held-out samples\n", n, V, nt, nv);
    printf("model    : %d -> %zu -> %zu -> %d   = %zu parameters\n", CTX * V, hidden, hidden, V, m.n_params);
    printf("training : SGD batch %zu, lr %g -> %g (cosine), %zu samples (%.2f passes)\n\n",
           batch, (double)lr, (double)(lr * lr_final), samples, (double)samples / (double)nt);
    printf("  chunk  samples      lr   train-loss  val-loss  val-acc   bits/char   samples/s\n");

    const size_t chunk = 500000;
    size_t chunks = (samples + chunk - 1) / chunk, cursor = nt;   /* force a shuffle first */
    double t_train = 0.0;
    for (size_t c = 1; c <= chunks; ++c) {
        float clr = ai_cosine_lr(lr, lr_final, c, chunks);
        double t0 = ai_now(), tl = 0.0; size_t seen = 0, in_batch = 0;
        while (seen < chunk) {
            if (cursor >= nt) {
                for (size_t i = nt - 1; i > 0; --i) { size_t j = ai_rng_below(&shuf, (uint32_t)(i + 1)); size_t t = trn[i]; trn[i] = trn[j]; trn[j] = t; }
                cursor = 0;
            }
            size_t p = trn[cursor++];
            encode_context(text, p, x);
            tl += ai_mlp_loss_grad(&m, x, text[p]);
            ++seen;
            if (++in_batch == batch) { ai_mlp_sgd_step(&m, clr, batch); in_batch = 0; }
        }
        if (in_batch) ai_mlp_sgd_step(&m, clr, in_batch);
        double dt = ai_now() - t0; t_train += dt;
        double vl, va; eval_val(&m, text, val, nv, 60000, x, &vl, &va);
        printf("  %5zu  %7zuk  %6.4f  %10.4f  %8.4f  %6.2f%%  %9.4f  %10.0f\n", c, c * chunk / 1000, (double)clr,
               tl / (double)seen, vl, va, vl / 0.693147180559945, (double)seen / dt);
        fflush(stdout);
    }
    double vl, va; eval_val(&m, text, val, nv, nv, x, &vl, &va);
    printf("\nfinal    : held-out loss %.4f nats/char (%.3f bits/char, perplexity %.2f), next-char accuracy %.2f%%\n",
           vl, vl / 0.693147180559945, exp(vl), va);
    printf("baseline : uniform guessing would be %.4f nats/char (perplexity %d)\n", log((double)V), V);
    printf("time     : %.1f s of training\n", t_train);
    if (save_model(out, &m)) { fprintf(stderr, "could not write %s\n", out); return 1; }
    printf("saved    : %s (%zu parameters)\n", out, m.n_params);
    return 0;
}

static int cmd_gen(const char *model, const char *prompt, size_t nchars, float temp, int topk, uint64_t seed)
{
    ai_mlp m;
    int r = load_model(model, &m, ai_select_backend(0));
    if (r) { fprintf(stderr, "cannot load model %s (%d)\n", model, r); return 1; }
    size_t cap = CTX + strlen(prompt) + nchars + 1;
    unsigned char *buf = malloc(cap);
    size_t len = CTX;
    memset(buf, 0, CTX);                       /* symbol 0 = space: left padding */
    for (const char *p = prompt; *p; ++p) { int s = sym_of(tolower((unsigned char)*p)); if (s >= 0) buf[len++] = (unsigned char)s; }
    float *x = malloc(sizeof(float) * CTX * V);
    ai_rng rng; ai_rng_seed(&rng, seed);
    size_t start = CTX;
    for (size_t i = 0; i < nchars; ++i) {
        encode_context(buf, len, x);
        const float *lg = ai_mlp_forward(&m, x);
        float sc[64]; int idx[64];
        for (int k = 0; k < V; ++k) { sc[k] = lg[k] / temp; idx[k] = k; }
        if (topk > 0 && topk < V) {            /* keep only the topk best */
            for (int a = 0; a < topk; ++a) for (int b = a + 1; b < V; ++b)
                if (sc[idx[b]] > sc[idx[a]]) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
        }
        int keep = (topk > 0 && topk < V) ? topk : V;
        float mx = sc[idx[0]]; for (int a = 1; a < keep; ++a) if (sc[idx[a]] > mx) mx = sc[idx[a]];
        float pr[64], z = 0.0f;
        for (int a = 0; a < keep; ++a) { pr[a] = expf(sc[idx[a]] - mx); z += pr[a]; }
        float u = ai_rng_uniform(&rng) * z, acc = 0.0f; int pick = idx[keep - 1];
        for (int a = 0; a < keep; ++a) { acc += pr[a]; if (u < acc) { pick = idx[a]; break; } }
        buf[len++] = (unsigned char)pick;
    }
    for (size_t i = start; i < len; ++i) putchar(ALPHABET[buf[i]]);
    putchar('\n');
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "train") && strcmp(argv[1], "gen"))) {
        fprintf(stderr, "usage: charlm train|gen [options]   (see the header of src/charlm.c)\n");
        return 2;
    }
    const char *corpus = "data/corpus.txt", *out = "data/philosophy.clm", *model = "data/philosophy.clm", *prompt = "";
    size_t hidden = 345, samples = 20000000, batch = 32, nchars = 600;
    float lr = 0.05f, lr_final = 0.05f, temp = 0.8f; int topk = 0; uint64_t seed = 1234;
    for (int i = 2; i < argc; ++i) {
        const char *a = argv[i]; int hv = i + 1 < argc;
        if      (!strcmp(a, "--corpus")   && hv) corpus  = argv[++i];
        else if (!strcmp(a, "--out")      && hv) out     = argv[++i];
        else if (!strcmp(a, "--model")    && hv) model   = argv[++i];
        else if (!strcmp(a, "--prompt")   && hv) prompt  = argv[++i];
        else if (!strcmp(a, "--hidden")   && hv) hidden  = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--samples")  && hv) samples = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--batch")    && hv) batch   = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--chars")    && hv) nchars  = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--seed")     && hv) seed    = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--topk")     && hv) topk    = atoi(argv[++i]);
        else if (!strcmp(a, "--lr")       && hv) lr      = (float)atof(argv[++i]);
        else if (!strcmp(a, "--lr-final") && hv) lr_final = (float)atof(argv[++i]);
        else if (!strcmp(a, "--temp")     && hv) temp    = (float)atof(argv[++i]);
        else { fprintf(stderr, "unknown or incomplete option: %s\n", a); return 2; }
    }
    if (!strcmp(argv[1], "gen")) {
        if (!(temp > 0.0f)) { fprintf(stderr, "temp must be > 0\n"); return 2; }
        return cmd_gen(model, prompt, nchars, temp, topk, seed);
    }
    if (hidden == 0 || batch == 0 || samples == 0 || !(lr > 0.0f) || !(lr_final > 0.0f && lr_final <= 1.0f)) {
        fprintf(stderr, "invalid training options\n"); return 2;
    }
    return cmd_train(corpus, out, hidden, samples, lr, lr_final, batch, seed);
}
