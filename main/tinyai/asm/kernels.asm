; =============================================================================
;  kernels.asm  --  AVX2 + FMA3 numeric kernels for tinyai
;  Assembler : NASM  (nasm -f win64)        Target ABI : Microsoft x64
; =============================================================================
;
;  unsigned ai_cpu_has_avx2_fma(void);
;  float    ai_dot_avx2 (const float *a, const float *b, size_t n);
;  void     ai_axpy_avx2(size_t n, float alpha, const float *x, float *y);
;
;  Microsoft x64 calling convention -- the parts that matter here
;    * Arguments are POSITIONAL: argument N lives in slot N whatever its type.
;      The slots are  RCX/XMM0, RDX/XMM1, R8/XMM2, R9/XMM3.  So for axpy the
;      count is in RCX, alpha in XMM1 (not XMM0!), x in R8 and y in R9.
;    * A float result is returned in XMM0.
;    * Volatile (free to clobber): RAX RCX RDX R8-R11, XMM0-XMM5, and the upper
;      halves of every YMM register.  RBX RBP RDI RSI R12-R15 and XMM6-XMM15 are
;      non-volatile: a function that wants them must save and restore them.
;
;  Rules every function in this file follows
;    1. Only volatile registers (ymm0..ymm5) are used.  There is no push/pop and
;       no stack frame, so each function is a true leaf and needs no SEH unwind
;       data (.pdata/.xdata).
;    2. Unaligned loads/stores only (vmovups), so plain malloc'd memory works.
;    3. VZEROUPPER before RET, so SSE-encoded code in the C runtime does not pay
;       AVX<->SSE transition penalties.
;    4. Memory outside [ptr, ptr + 4*n) is never read or written.  The tail is a
;       scalar loop instead of a masked load, so the kernels are safe to call on
;       a buffer that ends exactly at an unmapped page (tests/test_kernels.c
;       checks this with real guard pages).
; =============================================================================

default rel
bits 64

; CPUID.1:ECX bits: 12 = FMA, 27 = OSXSAVE, 28 = AVX
FEAT1_NEEDED equ (1 << 12) | (1 << 27) | (1 << 28)

global ai_cpu_has_avx2_fma
global ai_dot_avx2
global ai_axpy_avx2

section .text

; -----------------------------------------------------------------------------
;  unsigned ai_cpu_has_avx2_fma(void)
;
;  Returns 1 only if AVX2 *and* FMA can be used safely, i.e. the CPU has them
;  AND the operating system saves/restores YMM state on context switches.
;  Checking the CPUID bits alone is the classic mistake: a CPU can advertise AVX
;  while the OS has not enabled it, and the first YMM instruction then faults.
; -----------------------------------------------------------------------------
align 16
ai_cpu_has_avx2_fma:
    mov     r10, rbx                    ; CPUID clobbers RBX (non-volatile): park it
                                        ; in a volatile register, stay a leaf.
    xor     eax, eax
    cpuid                               ; leaf 0: EAX = highest basic leaf
    cmp     eax, 7
    jb      .no                         ; leaf 7 (AVX2 lives there) not available

    mov     eax, 1
    cpuid                               ; leaf 1: feature flags in ECX
    and     ecx, FEAT1_NEEDED
    cmp     ecx, FEAT1_NEEDED           ; need FMA + OSXSAVE + AVX, all three
    jne     .no

    xor     ecx, ecx
    xgetbv                              ; XCR0 -> EDX:EAX  (valid: OSXSAVE is set)
    and     eax, 6                      ; bit1 = SSE state, bit2 = AVX (YMM) state
    cmp     eax, 6
    jne     .no                         ; OS does not manage YMM registers

    mov     eax, 7
    xor     ecx, ecx
    cpuid                               ; leaf 7, sub-leaf 0
    bt      ebx, 5                      ; EBX bit 5 = AVX2
    jnc     .no

    mov     rbx, r10
    mov     eax, 1
    ret

.no:
    mov     rbx, r10
    xor     eax, eax
    ret


; -----------------------------------------------------------------------------
;  float ai_dot_avx2(const float *a, const float *b, size_t n)
;
;      return sum_{i<n} a[i] * b[i]
;
;  RCX = a, RDX = b, R8 = n.  Result in XMM0.
;
;  Four independent accumulators (ymm0-ymm3) hide the 4-cycle FMA latency, so
;  the main loop retires 32 floats per iteration.  ymm4/ymm5 hold the loads from
;  `a`; the loads from `b` are folded into the FMAs as memory operands.
;  The sum is re-associated (8 lanes x 4 accumulators), so the result can differ
;  from a plain left-to-right C loop in the last few bits.
; -----------------------------------------------------------------------------
align 16
ai_dot_avx2:
    vxorps      ymm0, ymm0, ymm0
    vxorps      ymm1, ymm1, ymm1
    vxorps      ymm2, ymm2, ymm2
    vxorps      ymm3, ymm3, ymm3

    cmp         r8, 32
    jb          .chunks8

.loop32:                                ; 32 floats per iteration
    vmovups     ymm4, [rcx]
    vmovups     ymm5, [rcx + 32]
    vfmadd231ps ymm0, ymm4, [rdx]       ; acc0 += a[0..7]   * b[0..7]
    vfmadd231ps ymm1, ymm5, [rdx + 32]  ; acc1 += a[8..15]  * b[8..15]
    vmovups     ymm4, [rcx + 64]
    vmovups     ymm5, [rcx + 96]
    vfmadd231ps ymm2, ymm4, [rdx + 64]  ; acc2 += a[16..23] * b[16..23]
    vfmadd231ps ymm3, ymm5, [rdx + 96]  ; acc3 += a[24..31] * b[24..31]
    add         rcx, 128
    add         rdx, 128
    sub         r8, 32
    cmp         r8, 32
    jae         .loop32

.chunks8:                               ; remaining whole groups of 8 floats
    cmp         r8, 8
    jb          .reduce
.loop8:
    vmovups     ymm4, [rcx]
    vfmadd231ps ymm0, ymm4, [rdx]
    add         rcx, 32
    add         rdx, 32
    sub         r8, 8
    cmp         r8, 8
    jae         .loop8

.reduce:                                ; 4 accumulators x 8 lanes -> 1 float
    vaddps      ymm0, ymm0, ymm1
    vaddps      ymm2, ymm2, ymm3
    vaddps      ymm0, ymm0, ymm2        ; 8 partial sums
    vextractf128 xmm1, ymm0, 1
    vaddps      xmm0, xmm0, xmm1        ; 4 partial sums  [s0 s1 s2 s3]
    vmovshdup   xmm1, xmm0              ; [s1 s1 s3 s3]
    vaddps      xmm0, xmm0, xmm1        ; lane0 = s0+s1, lane2 = s2+s3
    vmovhlps    xmm1, xmm1, xmm0        ; lane0 of xmm1 = lane2 of xmm0
    vaddss      xmm0, xmm0, xmm1        ; lane0 = (s0+s1) + (s2+s3)

    test        r8, r8                  ; 0..7 elements left: scalar loop
    jz          .done
.loop1:
    vmovss      xmm4, dword [rcx]
    vfmadd231ss xmm0, xmm4, dword [rdx]
    add         rcx, 4
    add         rdx, 4
    dec         r8
    jnz         .loop1

.done:
    vzeroupper                          ; XMM0 (the result) is unaffected
    ret


; -----------------------------------------------------------------------------
;  void ai_axpy_avx2(size_t n, float alpha, const float *x, float *y)
;
;      y[i] += alpha * x[i]      for i in [0, n)
;
;  RCX = n, XMM1 = alpha, R8 = x, R9 = y.   x and y must not overlap.
;
;  ymm0 holds alpha broadcast to all 8 lanes; ymm1..ymm4 carry the y vectors
;  (ymm1 is free to reuse once alpha has been broadcast out of xmm1).
; -----------------------------------------------------------------------------
align 16
ai_axpy_avx2:
    vbroadcastss ymm0, xmm1             ; ymm0 = alpha x8

    cmp         rcx, 32
    jb          .chunks8

.loop32:                                ; 32 floats per iteration
    vmovups     ymm1, [r9]
    vmovups     ymm2, [r9 + 32]
    vmovups     ymm3, [r9 + 64]
    vmovups     ymm4, [r9 + 96]
    vfmadd231ps ymm1, ymm0, [r8]        ; y += alpha * x
    vfmadd231ps ymm2, ymm0, [r8 + 32]
    vfmadd231ps ymm3, ymm0, [r8 + 64]
    vfmadd231ps ymm4, ymm0, [r8 + 96]
    vmovups     [r9],      ymm1
    vmovups     [r9 + 32], ymm2
    vmovups     [r9 + 64], ymm3
    vmovups     [r9 + 96], ymm4
    add         r8, 128
    add         r9, 128
    sub         rcx, 32
    cmp         rcx, 32
    jae         .loop32

.chunks8:
    cmp         rcx, 8
    jb          .tail
.loop8:
    vmovups     ymm1, [r9]
    vfmadd231ps ymm1, ymm0, [r8]
    vmovups     [r9], ymm1
    add         r8, 32
    add         r9, 32
    sub         rcx, 8
    cmp         rcx, 8
    jae         .loop8

.tail:                                  ; 0..7 elements left
    test        rcx, rcx
    jz          .done
.loop1:
    vmovss      xmm1, dword [r9]
    vfmadd231ss xmm1, xmm0, dword [r8]  ; xmm0 lane 0 still holds alpha
    vmovss      dword [r9], xmm1
    add         r8, 4
    add         r9, 4
    dec         rcx
    jnz         .loop1

.done:
    vzeroupper
    ret
