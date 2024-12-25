function compile_with_tile_size() {
    local tile_size=$1
    nvcc -arch=sm_75 -L/usr/local/cuda/lib64 -lcublas ./matrix_mul.cu -DTILE_SIZE=$tile_size -o $tile_size
    ./$tile_size 0 1000
}

for tile_size in 32 64 128; do
    echo "tile_size: $tile_size"
    compile_with_tile_size $tile_size
done
