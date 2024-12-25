#!/bin/bash

# 定义函数：编译代码
function compile_with_tile_size() {
    local tile_size=$1  # 块大小
    echo "Compiling with TILE_SIZE=$tile_size"
    nvcc -w -arch=sm_75 -L/usr/local/cuda/lib64 ./matrix_mul_lab4.cu -DTILE_SIZE=$tile_size -o ./a.out

    # 检查编译是否成功
    if [[ $? -ne 0 ]]; then
        echo "Error: Compilation failed for TILE_SIZE=$tile_size"
        exit 1
    fi
}

# 定义函数：运行测试
function run_tests() {
    local tile_size=$1  # 当前块大小

    # 矩阵大小的数组
    local sizes=(100 200 400 800 1000)
    echo "======================================="
    echo "Running tests with TILE_SIZE=$tile_size"
    echo "======================================="

    for size in "${sizes[@]}"; do
        echo "Matrix size: $size"
        ./a.out 1 $size  # 默认启用结果检查
        echo "--------------------------------"
        echo
    done
}

# 主脚本逻辑
echo "---****** Starting Lab4 Tests ******---"
for tile_size in 32 64 128; do  # 循环测试不同的 TILE_SIZE
    compile_with_tile_size $tile_size   # 编译
    run_tests $tile_size                # 运行测试
done

echo "---****** All Lab4 Tests Completed ******---"
