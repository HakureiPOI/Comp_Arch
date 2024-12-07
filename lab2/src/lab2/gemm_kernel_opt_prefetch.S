.text;
.p2align 2;
.global gemm_kernel_opt_prefetch;
.type gemm_kernel_opt_prefetch, %function;

#define     MAT_C               %rdi
#define     MAT_A               %rsi
#define     MAT_B               %r14
#define     DIM_M               %rcx
#define     DIM_N               %r8
#define     DIM_K               %r9
#define     loop_m              %r10
#define     loop_k              %r11
#define     loop_n              %r12
#define     mat_elem_idx        %r13
#define     prefetch_elem_idx   %r15


.macro PUSHD   // 保存原通用寄存器值
    push %rax
    push %rbx
    push %rcx
    push %rdx
    push %rsi
    push %rdi
    push %rbp
    push %r8
    push %r9
    push %r10
    push %r11
    push %r12
    push %r13
    push %r14
    push %r15
.endm

.macro POPD    // 恢复原通用寄存器值
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %r11
    pop %r10
    pop %r9
    pop %r8
    pop %rbp
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %rbx
    pop %rax
.endm

.macro GEMM_INIT
    mov %rdx, MAT_B

    xor loop_m, loop_m
    xor loop_k, loop_k
    xor loop_n, loop_n
.endm

.macro DO_GEMM
    xor loop_k, loop_k  // 初始化 k 循环

DO_LOOP_K:
    xor loop_m, loop_m  // 初始化 m 循环

DO_LOOP_M:
    xor loop_n, loop_n  // 初始化 n 循环

    // 加载 A[m][k] 并计算索引
    mov loop_m, %rax
    imul DIM_K, %rax                   // mat_elem_idx = m * DIM_K
    add loop_k, %rax                   // mat_elem_idx += k
    mov %rax, mat_elem_idx
    vmovss (MAT_A, mat_elem_idx, 4), %xmm0  // 加载 A[m][k] 到 XMM 寄存器

    // 预取 A[m + 1][k]
    lea (mat_elem_idx, DIM_K), %rbx
    prefetchnta (MAT_A, %rbx, 4)

DO_LOOP_N:
    // 加载 B[k][n] 并计算索引
    mov loop_k, %rax
    imul DIM_N, %rax                   // mat_elem_idx = k * DIM_N
    add loop_n, %rax                   // mat_elem_idx += n
    mov %rax, mat_elem_idx
    vmovss (MAT_B, mat_elem_idx, 4), %xmm1  // 加载 B[k][n] 到 XMM 寄存器

    // 预取 B[k][n + 1]
    lea (mat_elem_idx, 4), %rbx        // 每元素4字节
    prefetchnta (MAT_B, %rbx, 4)

    // C[m][n] = A[m][k] * B[k][n] + C[m][n]
    mov loop_m, %rax
    imul DIM_N, %rax                   // mat_elem_idx = m * DIM_N
    add loop_n, %rax                   // mat_elem_idx += n
    mov %rax, mat_elem_idx
    vmovss (MAT_C, mat_elem_idx, 4), %xmm2  // 加载 C[m][n] 到 XMM 寄存器
    vfmadd231ss %xmm0, %xmm1, %xmm2        // xmm2 = xmm0 * xmm1 + xmm2
    vmovss %xmm2, (MAT_C, mat_elem_idx, 4) // 存储结果到 C[m][n]

    // 预取 C[m][n]
    prefetchnta (MAT_C, mat_elem_idx, 4)

    // 更新 n 循环
    add $1, loop_n
    cmp DIM_N, loop_n
    jl DO_LOOP_N

    // 更新 m 循环
    add $1, loop_m
    cmp DIM_M, loop_m
    jl DO_LOOP_M

    // 更新 k 循环
    add $1, loop_k
    cmp DIM_K, loop_k
    jl DO_LOOP_K
.endm


gemm_kernel_opt_prefetch:
    PUSHD
    GEMM_INIT
    DO_GEMM
    POPD
    ret
    

