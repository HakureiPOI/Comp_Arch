/**
 *  向标准输出打印1个64位整数
 * */

.section .bss
//    .lcomm num, 8       // 存储64位整数
    .lcomm buffer, 21   // 20个数字 + 1个空字符 作为输出的缓冲区

.section .data
    newline: .byte 0xA      // 换行符

.section .text
    .globl _start

_start:
    // 初始化要打印的数字
    mov $1234567890123456789, %rax
//    mov %rax, num(%rip)

    // 将整数转换为字符串
//    mov num(%rip), %rax
    lea buffer+20(%rip), %rdi   // 将输出字符串的最后一个字符地址放入rdi寄存器
    movb $0, (%rdi)             // 给取后一个字符赋'\0'标志结束

convert_loop:
    xor     %rdx, %rdx          // 清空 %rdx，准备存储除法的余数
    mov     $10, %rcx           // 除数设置为 10
    div     %rcx                // rdx = rax % 10, rax = rax / 10
    add     $'0', %dl           // 将余数转换为对应的 ASCII 字符
    dec     %rdi                // 准备写入前一个内存位置
    mov     %dl, (%rdi)         // 将字符写入目标内存
    test    %rax, %rax          // 检查 rax 是否为 0
    jnz     convert_loop        // 如果 rax 不为 0，则继续循环


find_start:                     // 转换结束，跳过字符串前的所有0
    cmpb $'0', (%rdi)
    jne print_string
    inc %rdi
    jmp find_start

print_string:                   // 开始输出字符串
                                // 计算字符串长度
    lea buffer+20(%rip), %rax
    sub %rdi, %rax              // 计算存储的字节数
    mov %rax, %rdx              // 待输出的字节数放入rdx中

    // 系统调用号 (sys_write)     // TODO: 请修复此处的系统调用以正常输出字符串
    mov      $1, %rax // syscall number
    //       指向字符串的指针
    mov      %rdi, %rsi
    //       文件描述符 (stdout)
    mov      $1, %rdi
    //       要写入的字节数
    mov      %rdx, %rdx
    //       执行系统调用
    syscall

    // 打印换行符
    mov $1, %rax
    mov $1, %rdi
    lea newline(%rip), %rsi
    mov $1, %rdx
    syscall

    // 退出程序
    mov $60, %rax  // 系统调用号 (sys_exit)
    xor %rdi, %rdi // 退出状态码
    syscall