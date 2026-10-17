; znver3_ke.s
; Kinetic energy on the host. The 3090 Ti should not be asked to do this
; as a scalar tail — solids are already zeroed on the device, so a flat
; AVX2 dot product on the 5900X is the whole reduction.
;
;   float ke_fluid(const float *u, const float *v, int n);
;   System V AMD64: rdi = u, rsi = v, rdx = n, result in xmm0.
;
; Zen 3 (znver3): vfmadd231ps is the right instruction. vzeroupper before
; ret keeps the upper YMM state from taxing a later SSE caller.
; Assemble: clang -c -o znver3_ke.o znver3_ke.s

.intel_syntax noprefix
.section .text
.globl ke_fluid
.type ke_fluid, @function
.align 16
ke_fluid:
    vxorps ymm0, ymm0, ymm0
    xor rax, rax
    cmp rdx, 8
    jl .horiz

.align 16
.loop:
    vmovups ymm1, ymmword ptr [rdi + rax*4]
    vfmadd231ps ymm0, ymm1, ymm1
    vmovups ymm2, ymmword ptr [rsi + rax*4]
    vfmadd231ps ymm0, ymm2, ymm2
    add rax, 8
    cmp rax, rdx
    jb .loop

.horiz:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper

.tail:
    cmp rax, rdx
    jae .half
    movss xmm1, dword ptr [rdi + rax*4]
    mulss xmm1, xmm1
    addss xmm0, xmm1
    movss xmm1, dword ptr [rsi + rax*4]
    mulss xmm1, xmm1
    addss xmm0, xmm1
    inc rax
    jmp .tail

.half:
    mov eax, 0x3f000000
    movd xmm1, eax
    mulss xmm0, xmm1
    ret
.size ke_fluid, .-ke_fluid

; C side, so the 12 cores can split the y-range while the GPU is in the
; field kernel. Compile with: gcc -O3 -march=znver3 -c host_reduce.c
;
;   #include <stddef.h>
;   extern float ke_fluid(const float *u, const float *v, int n);
;
;   float ke_rows(const float *u, const float *v, int nx, int y0, int y1) {
;       return ke_fluid(u + y0 * nx, v + y0 * nx, (y1 - y0) * nx);
;   }
