function compile() {
    local use_cublas=$1
    if [ $use_cublas -eq 1 ]; then
        echo "use cublas"
        nvcc -w -arch=sm_75 -L/usr/local/cuda/lib64 -lcublas ./matrix_mul_lab5.cu -o ./a.out
    else
        echo "use no cublas"
        nvcc -w -arch=sm_75 -L/usr/local/cuda/lib64 ./matrix_mul_lab5.cu -o ./a.out
    fi
}

function test() {
    for matrix_size in 100 300 500 1000 ; do
        echo "matrix_size: $matrix_size"
        ./a.out 0 $matrix_size
        echo "--------------------------------"
        echo 
    done
}

echo "Testing with cuBLAS"
compile 1
test

echo "Testing without cuBLAS"
compile 0
test
