# tinyai — a neural network in C, x86-64 assembly and Common Lisp

A small neural network, written from scratch with no libraries. The model, the
training loop and the data are C. The two loops that dominate the run time are
hand-written **AVX2 + FMA assembly** (NASM, Windows x64 ABI), picked at run time
after a proper CPUID/XGETBV check, with a plain-C fallback. A **Common Lisp**
front end (SBCL) drives the very same compiled code through its foreign-function
interface.

It is a classifier, not a chatbot: the demo trains a `2 → 64 → 64 → 3` ReLU network
(4,547 parameters) to separate three interleaved spiral arms, a task a linear model
cannot do.

Real output of `.\build\tinyai.exe --no-plot --log-every 100` (the `test%` column is
held-out points; the speed line varies ~10 % from run to run):

```
 epoch        lr       loss    train%     test%
     1   0.02000    0.97871    42.83%    38.83%
   100   0.01534    0.05150    98.33%    96.33%
   200   0.00584    0.00806    99.67%    98.50%
   300   0.00100    0.00575    99.83%    98.33%

final   : train 99.83%  test 98.33%  (test loss 0.0624)
speed   : 0.242 s of pure training, 742573 samples/s  [AVX2+FMA (NASM)]
```

Without `--no-plot` it also prints an ASCII map of the decision regions.

## Quick start

Needs MSYS2 UCRT64 `gcc` and `nasm` (SBCL is optional, for the Lisp part). The build
script adds `C:\msys64\ucrt64\bin` to `PATH` if `gcc` is not already on it. There is
no `make`; everything is one PowerShell script.

```powershell
cd "C:\User 0\docs\tinyai"
.\build.ps1 -Test -Lisp -Run     # build, run all three test suites, then the demo

.\build\tinyai.exe --help        # every option (epochs, lr, hidden width, noise, ...)
.\build\tinyai.exe --scalar      # the same program on the plain-C kernels
.\build\bench.exe                # kernel + end-to-end benchmarks
sbcl --noinform --non-interactive --load lisp\spiral.lisp     # the demo, driven from Lisp
sbcl --noinform --load lisp\tinyai.lisp --eval "(in-package :tinyai)"   # a REPL
```

## How it works

```
      Common Lisp (lisp/tinyai.lisp)            C command line (src/main.c)
              │  sb-alien FFI                              │
              ▼                                            ▼
       build/tinyai.dll ◄── src/tinyai_api.c         src/train.c   epoch loop, evaluation, LR schedule
                                  └──────────────────────┤
                                                    src/mlp.c      forward pass, back-propagation, SGD
                                                         │  (table of function pointers: ai_backend)
                                       ┌─────────────────┴──────────────────┐
                                src/kernels.c                        asm/kernels.asm
                                dot, axpy in plain C                 dot, axpy in AVX2+FMA, CPUID check
```

Every weight matrix is stored **row-major, `out × in`**, so each neuron's weights are
one contiguous row. That reduces the entire network to two kernels:

| step | work | kernel |
|---|---|---|
| forward | `z[j] = W_j · input + b[j]` | one `dot` per neuron |
| backward, input gradient | `dprev += δ[j] · W_j` | one `axpy` per neuron |
| backward, weight gradient | `gW_j += δ[j] · input` | one `axpy` per neuron |
| update | `params -= lr · grads` | one long `axpy` over *all* parameters |

Softmax, the loss, ReLU and the learning-rate schedule stay in C: they are a few
percent of the time. Gradients are accumulated, not overwritten, so a mini-batch is
just several backward passes before one update.

## The assembly (`asm/kernels.asm`)

* `ai_cpu_has_avx2_fma()` — checks the CPUID bits **and** `XGETBV` (the OS must save
  YMM state, otherwise the first YMM instruction faults). `RBX`, which `CPUID`
  clobbers, is parked in `R10` so the function stays a leaf.
* `ai_dot_avx2` — four independent accumulators hide the 4-cycle FMA latency; 32
  floats per iteration; horizontal sum; scalar tail.
* `ai_axpy_avx2` — alpha is broadcast; 32 floats per iteration; scalar tail. Under the
  Win64 ABI arguments are positional, so `alpha` (the 2nd argument) arrives in `XMM1`.
* Rules every function follows: only volatile registers (`ymm0–ymm5`), so no stack
  frame and no SEH unwind data; unaligned loads only; `vzeroupper` before `ret`; never
  touch memory outside `[ptr, ptr + 4n)` (the tail is a scalar loop, not a masked load).

## How it is tested

| suite | what it proves | checks |
|---|---|---|
| `test_kernels.exe` | asm == C reference == double-precision reference, for 82 sizes (every tail length) | 6,130 |
| ↳ guard pages | buffers end exactly at an unmapped page: one stray byte read or written crashes the test | |
| ↳ ABI harness | `RBX RBP RDI RSI R12–R15 XMM6–15` survive every call (canaries in all of them) | |
| `test_gradcheck.exe` | back-propagation == finite differences of an independent double-precision forward pass, every weight and bias | 249 |
| `lisp/test.lisp` | Lisp wrappers, argument validation, handle lifetime, and **Lisp-driven training == the C program** (same accuracies, same loss, same ASCII picture cell for cell) | 538 |

Passing tests only matter if they can fail, so I injected bugs and checked that each
was caught. Seven assembly bugs (dropped tail, a dead 32-byte over-read, an off-by-one
write, `alpha` read from the wrong register, a lost accumulator, a clobbered `RBX`, an
unsaved `XMM6`) and five Lisp-side bugs (wrong FFI float type, swapped arguments, a
perturbed learning-rate schedule, a removed fit check, a wrong label) — all caught.
The first version of the suite *missed* the clobbered `RBX`; that gap is why the ABI
harness exists.

Scalar and AVX2 training also agree statistically: over 8 seeds both reach a mean of
99.94 % train / 98.42 % test. They are not bit-identical (FMA and re-association round
differently, and SGD amplifies that), so the loss curves drift apart mid-training and
meet again as the learning rate decays.

## Performance (Intel Core 5 210H, gcc 16.2, NASM 3.02)

Measured on a pinned performance core, variants timed round-robin, 64-byte-aligned
buffers; best of 11 rounds. GFLOP/s, higher is better. (Selected sizes shown;
`bench.exe` prints seven per kernel. Numbers move a few percent from run to run.)

| n | dot: scalar C | dot: gcc `-O3` | **dot: asm** | axpy: scalar C | axpy: gcc `-O3` | **axpy: asm** |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9.7 | 50.3 | **44.0** | 5.4 | 39.4 | **32.3** |
| 1,024 | 5.1 | 88.7 | **90.9** | 7.7 | 63.3 | **65.5** |
| 4,096 | 4.8 | 81.5 | **81.5** | 7.8 | 65.3 | **71.3** |
| 1,048,576 (memory-bound) | 3.7 | 13.2 | **12.3** | 5.6 | 9.8 | **9.1** |

("gcc `-O3`" is the same C loop built with `-O3 -march=native -ffast-math -funroll-loops`.)

End-to-end training, one sample per step:

| network | scalar C | AVX2 asm | speed-up |
|---|---:|---:|---:|
| 2-64-64-3 (the demo) | 156,000 samples/s | 722,000 | **4.6×** |
| 784-128-10 (MNIST-sized) | 10,700 | 50,200 | **4.7×** |
| 784-256-128-10 | 4,370 | 13,300 | **3.0×** |

What to take from it:

* The assembly is up to **~18× faster than the plain C** fallback on in-cache vectors
  (dot, n = 1,024) and 1.6–3.3× when memory-bound.
* It is **not faster than what gcc generates when allowed every optimisation**: across
  all 14 measurements (`bench.exe` prints every size) the ratio asm/gcc ranges from
  0.82× to 1.35× with a geometric mean of about 1.02×, i.e. a tie. The point of the
  assembly here is that it does not depend on `-march=native -ffast-math`, it
  dispatches at run time, and it is a worked example of the Win64 ABI and AVX2. It is
  not a win over a good compiler.
* End-to-end gains are smaller than kernel gains because softmax, ReLU, call overhead
  and short rows take a fixed share (Amdahl).
* Benchmarking lesson: my first `bench.exe` said the asm `axpy` was 2.7× *slower* than
  gcc at n = 1024. It was not; the thread had been moved to an efficiency core. Pinning,
  round-robin timing and aligned buffers fixed the harness, not the kernel. A 32-byte
  access that straddles a cache line costs ~40 % throughput, for asm and gcc alike,
  which is why the network's buffers are 64-byte aligned.

## The Lisp layer (`lisp/`)

SBCL loads `build\tinyai.dll` and wraps it in ordinary Lisp. Single-float vectors go to
the assembly **without copying** (they are pinned, not marshalled); C owns the memory,
Lisp owns the lifetime (`with-data`, `with-net`, plus finalizers as a REPL safety net).

```lisp
(load "lisp/tinyai.lisp") (in-package :tinyai)

(dot '(1 2 3 4) '(5 6 7 8))                       ; => 70.0, computed by the assembly

(with-data ((train (make-spirals :seed 1234))
            (test  (make-spirals :seed 1235)))
  (with-net ((net (make-net '(2 64 64 3) :seed 1236)))
    (fit net train :epochs 300 :test test)        ; prints the table above
    (plot-regions net test)))
```

`lisp/spiral.lisp` treats an experiment as a plain list, so ordinary list functions can
copy and sweep it. One sweep (120 epochs each):

```
  hidden    params   train%    test%   seconds
       2        21    33.33    33.33     0.006     <- too narrow to learn the spiral
       4        47    70.50    69.33     0.009
       8       123    98.00    95.83     0.012
      16       371    99.50    98.00     0.021
      32     1,251   100.00    98.17     0.038
      64     4,547   100.00    98.17     0.099
```

Gotchas handled for you: SBCL traps float overflow/invalid by default, so every foreign
call runs with traps masked; C functions return NaN or `NULL` on bad input and the
Lisp wrappers turn that into a `tinyai-error` instead of a crash.

## Layout

```
build.ps1               build + test script (no make needed)
asm/kernels.asm         the AVX2+FMA kernels and the CPUID check
asm/abi_check.asm       test-only: canary registers to verify the Win64 ABI
src/kernels.[ch]        kernel API, scalar C reference, backend selection
src/mlp.[ch]            the network: forward, backward, SGD
src/train.[ch]          epoch loop, evaluation, learning-rate schedule
src/data.[ch]  rng.h  timer.h
src/main.c              command-line demo            -> build\tinyai.exe
src/tinyai_api.[ch]     flat C API for other languages -> build\tinyai.dll
src/test_kernels.c  test_gradcheck.c  bench.c  bench_cnative.c
lisp/tinyai.lisp  spiral.lisp  test.lisp
```

## Limitations

* **Windows x64 only.** The assembly follows the Microsoft x64 ABI. Linux/macOS use
  different argument registers and would need a small port of the three functions.
* The assembly path needs **AVX2 + FMA** (any Intel since Haswell, AMD since Zen);
  otherwise the scalar C path is used automatically. There is no AVX-512 variant.
* Single-threaded, float32, fully-connected layers only, plain SGD (no momentum/Adam).
  There is no batched matrix-multiply kernel, so it suits small networks.
* It does not read real datasets yet; the spiral data is generated.

## Where to go next

1. **Real data.** An MNIST loader and a `784-128-10` network (the benchmark already
   trains that shape at ~50,000 samples/s; expect roughly 97 % accuracy).
2. **A GEMM micro-kernel in assembly** (e.g. 6×16 with 12 accumulators — which needs
   proper SEH unwind info for `XMM6–15`) for fast mini-batch training.
3. **A tiny character-level text model**, to make it *generate* something.
4. **Lisp writing the assembly**: macros that emit the NASM kernels for any unroll
   factor, or an AVX-512 variant.
5. Adam, momentum, dropout, more layers.
