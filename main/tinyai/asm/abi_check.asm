; =============================================================================
;  abi_check.asm  --  TEST-ONLY helper (linked into test_kernels.exe, nothing else)
;
;  unsigned abi_check(size_t fn, size_t a, size_t b, size_t c, size_t d, float alpha);
;
;  Loads a distinct canary into every NON-VOLATILE register of the Microsoft x64
;  ABI (RBX RBP RDI RSI R12-R15 and XMM6-XMM15), calls  fn  with
;        RCX = a   RDX = b   R8 = c   R9 = d   XMM1 = alpha
;  and returns a bitmask of the registers that did not survive the call:
;        bit 0 RBX   1 RBP   2 RDI   3 RSI   4..7 R12..R15   8..17 XMM6..XMM15
;  0 means fn honoured the ABI.  (How a kernel's arguments map to a..d / alpha:
;        dot (a, b, n)          ->  a = a, b = b, c = n
;        axpy(n, alpha, x, y)   ->  a = n, c = x, d = y, alpha = alpha
;        ai_cpu_has_avx2_fma()  ->  all zero.)
;
;  abi_selftest_clobber() deliberately breaks the rules (RBX, R12, XMM6) so the
;  test can prove that this harness really does notice a violation.
; =============================================================================

default rel
bits 64

global abi_check
global abi_selftest_clobber

; Offsets of abi_check's own arguments once its frame is built:
; entry rsp = rsp + 8 pushes (64) + 200 bytes of locals = rsp + 264.
%define ARG_FN     272          ; [entry + 8]   (home slot of rcx)
%define ARG_A      280          ; [entry + 16]
%define ARG_B      288          ; [entry + 24]
%define ARG_C      296          ; [entry + 32]
%define ARG_D      304          ; [entry + 40]  (5th argument, on the stack)
%define ARG_ALPHA  312          ; [entry + 48]  (6th argument, a float in the low 4 bytes)

%macro SETXMM 2                 ; xmm index, 64-bit canary
    mov         rax, %2
    movq        xmm %+ %1, rax
    punpcklqdq  xmm %+ %1, xmm %+ %1
%endmacro

%macro CHKREG 3                 ; register, canary, result bit
    mov         r11, %2
    cmp         %1, r11
    je          %%ok
    or          r10d, 1 << %3
%%ok:
%endmacro

%macro CHKXMM 3                 ; xmm index, 64-bit canary, result bit
    mov         rax, %2
    movq        xmm0, rax
    punpcklqdq  xmm0, xmm0
    pcmpeqb     xmm0, xmm %+ %1
    pmovmskb    eax, xmm0
    cmp         eax, 0xFFFF
    je          %%ok
    or          r10d, 1 << %3
%%ok:
%endmacro

section .text

align 16
abi_check:
    mov     [rsp + 8],  rcx             ; park our arguments in the caller-provided
    mov     [rsp + 16], rdx             ; shadow space: the canaries are about to
    mov     [rsp + 24], r8              ; overwrite every register we could keep
    mov     [rsp + 32], r9              ; them in
    push    rbx
    push    rbp
    push    rdi
    push    rsi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 200                    ; 32 shadow + 160 (xmm6-15) + 8: rsp 16-aligned

    movups  [rsp + 32],  xmm6           ; keep OUR caller's xmm6-15 safe
    movups  [rsp + 48],  xmm7
    movups  [rsp + 64],  xmm8
    movups  [rsp + 80],  xmm9
    movups  [rsp + 96],  xmm10
    movups  [rsp + 112], xmm11
    movups  [rsp + 128], xmm12
    movups  [rsp + 144], xmm13
    movups  [rsp + 160], xmm14
    movups  [rsp + 176], xmm15

    mov     rbx, 0x1111111111111111
    mov     rbp, 0x2222222222222222
    mov     rdi, 0x3333333333333333
    mov     rsi, 0x4444444444444444
    mov     r12, 0x5555555555555555
    mov     r13, 0x6666666666666666
    mov     r14, 0x7777777777777777
    mov     r15, 0x8888888888888888
    SETXMM  6,  0xC0DEC0DEC0DE0006
    SETXMM  7,  0xC0DEC0DEC0DE0007
    SETXMM  8,  0xC0DEC0DEC0DE0008
    SETXMM  9,  0xC0DEC0DEC0DE0009
    SETXMM  10, 0xC0DEC0DEC0DE000A
    SETXMM  11, 0xC0DEC0DEC0DE000B
    SETXMM  12, 0xC0DEC0DEC0DE000C
    SETXMM  13, 0xC0DEC0DEC0DE000D
    SETXMM  14, 0xC0DEC0DEC0DE000E
    SETXMM  15, 0xC0DEC0DEC0DE000F

    mov     rax, [rsp + ARG_FN]
    mov     rcx, [rsp + ARG_A]
    mov     rdx, [rsp + ARG_B]
    mov     r8,  [rsp + ARG_C]
    mov     r9,  [rsp + ARG_D]
    movss   xmm1, [rsp + ARG_ALPHA]
    call    rax                         ; ---- the function under test ----

    xor     r10d, r10d                  ; result mask (r10/r11/rax/xmm0 are volatile,
    CHKREG  rbx, 0x1111111111111111, 0  ;  so using them after the call is fine)
    CHKREG  rbp, 0x2222222222222222, 1
    CHKREG  rdi, 0x3333333333333333, 2
    CHKREG  rsi, 0x4444444444444444, 3
    CHKREG  r12, 0x5555555555555555, 4
    CHKREG  r13, 0x6666666666666666, 5
    CHKREG  r14, 0x7777777777777777, 6
    CHKREG  r15, 0x8888888888888888, 7
    CHKXMM  6,  0xC0DEC0DEC0DE0006, 8
    CHKXMM  7,  0xC0DEC0DEC0DE0007, 9
    CHKXMM  8,  0xC0DEC0DEC0DE0008, 10
    CHKXMM  9,  0xC0DEC0DEC0DE0009, 11
    CHKXMM  10, 0xC0DEC0DEC0DE000A, 12
    CHKXMM  11, 0xC0DEC0DEC0DE000B, 13
    CHKXMM  12, 0xC0DEC0DEC0DE000C, 14
    CHKXMM  13, 0xC0DEC0DEC0DE000D, 15
    CHKXMM  14, 0xC0DEC0DEC0DE000E, 16
    CHKXMM  15, 0xC0DEC0DEC0DE000F, 17

    movups  xmm6,  [rsp + 32]
    movups  xmm7,  [rsp + 48]
    movups  xmm8,  [rsp + 64]
    movups  xmm9,  [rsp + 80]
    movups  xmm10, [rsp + 96]
    movups  xmm11, [rsp + 112]
    movups  xmm12, [rsp + 128]
    movups  xmm13, [rsp + 144]
    movups  xmm14, [rsp + 160]
    movups  xmm15, [rsp + 176]
    mov     eax, r10d
    add     rsp, 200
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rdi
    pop     rbp
    pop     rbx
    ret

align 16
abi_selftest_clobber:                   ; breaks the ABI on purpose: RBX, R12, XMM6
    mov     rbx, 1
    mov     r12, 2
    xorps   xmm6, xmm6
    ret
