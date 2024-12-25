#!/bin/bash

# 定义参数数组
sizes=(100 200 400 800)

# 循环执行每个命令
for size in "${sizes[@]}"; do
    echo "Running with input size: $size"
    ./a.out 1 $size
    echo "--------------------------------"
    echo # 添加空行以分隔不同结果
done
