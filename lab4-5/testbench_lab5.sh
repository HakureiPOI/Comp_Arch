function compile() {
    local use_cublas=$1
    local tile_width=$2  # 添加块大小参数
    if [ $use_cublas -eq 1 ]; then
        echo "use cublas with TILE_WIDTH=$tile_width"
        echo
        nvcc -w -arch=sm_75 -L/usr/local/cuda/lib64 -lcublas -DTILE_WIDTH=$tile_width ./matrix_mul_lab5.cu -o ./a.out
    else
        echo "use no cublas with TILE_WIDTH=$tile_width"
        echo
        nvcc -w -arch=sm_75 -L/usr/local/cuda/lib64 -DTILE_WIDTH=$tile_width ./matrix_mul_lab5.cu -o ./a.out
    fi

    # 检查编译是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed for TILE_WIDTH=$tile_width"
        exit 1
    fi
}

function test() {
    for tile_width in 16 32 64 128; do  # 测试不同块大小
        echo "=======================================" | tee -a results.log
        echo "Testing with TILE_WIDTH=$tile_width" | tee -a results.log
        echo "=======================================" | tee -a results.log
        compile $1 $tile_width  # 传入当前的块大小
        for matrix_size in 128 256 512 1024 2048; do  # 测试不同矩阵大小
            echo "matrix_size: $matrix_size, TILE_WIDTH: $tile_width" | tee -a results.log
            ./a.out 0 $matrix_size | tee -a results.log
            echo "--------------------------------" | tee -a results.log
            echo | tee -a results.log
        done
    done
}

echo "---****** Testing with cuBLAS ******---" | tee -a results.log
echo | tee -a results.log
test 1  # 使用 cuBLAS 测试

echo "---****** Testing without cuBLAS ******---" | tee -a results.log
echo | tee -a results.log
test 0  # 不使用 cuBLAS 测试
