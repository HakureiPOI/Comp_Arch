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
DO_LOOP_K:
    xor loop_m, loop_m

DO_LOOP_M:
    xor loop_n, loop_n

    // 计算 A[m][k] 的索引和加载
    mov loop_m, %rax
    imul DIM_K, %rax
    add loop_k, %rax                         // mat_elem_idx = m*K + k
    mov %rax, mat_elem_idx
    flds (MAT_A, mat_elem_idx, 4)            // 加载 A[m][k]

    // 预取下一行 A
    lea (mat_elem_idx, DIM_K), %rbx          // mat_elem_idx + DIM_K
    prefetchnta (MAT_A, %rbx, 4)             // prefetch A[m + 1][k]

DO_LOOP_N:
    // 计算 B[k][n] 的索引和加载
    mov loop_k, %rax
    imul DIM_N, %rax
    add loop_n, %rax                         // mat_elem_idx = k*N + n
    mov %rax, mat_elem_idx
    flds (MAT_B, mat_elem_idx, 4)            // 加载 B[k][n]

    // 预取下一列 B
    mov mat_elem_idx, %rbx
    add $4, %rbx                             // 每元素4字节
    prefetchnta (MAT_B, %rbx, 4)             // prefetch B[k][n + 1]

    // 计算 A[m][k] * B[k][n]
    fmul %st(1), %st(0)                      // A[m][k] * B[k][n]

    // 计算 C[m][n] 的索引、加载、计算和存储
    mov loop_m, %rax
    imul DIM_N, %rax
    add loop_n, %rax                         // mat_elem_idx = m*N + n
    mov %rax, mat_elem_idx
    flds (MAT_C, mat_elem_idx, 4)            // 加载 C[m][n]

    // 预取下一步的 C
    prefetchnta (MAT_C, mat_elem_idx, 4)     // prefetch C[m][n]

    faddp %st(0), %st(1)                     // C[m][n] += A[m][k] * B[k][n]
    fstps (MAT_C, mat_elem_idx, 4)           // 存储结果到 C[m][n]

    // 内层循环迭代
    inc loop_n
    cmp DIM_N, loop_n
    jl DO_LOOP_N

    fstp %st(0)                              // 弹出浮点堆栈元素
    add $1, loop_m
    cmp DIM_M, loop_m
    jl DO_LOOP_M

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
    

