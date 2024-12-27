import subprocess
import re
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run models with or without optimization.")
parser.add_argument(
    "--use-optim",
    action="store_true",
    help="Enable optimization for running models (default is without optimization)."
)
args = parser.parse_args()

# 模型文件列表
model_files = ["stories15M.bin", "stories42M.bin", "stories110M.bin"]

# 每个模型运行的次数
num_runs = 5

# 用于存储运行结果
results = {}

# 根据标志位选择运行命令
executable = "./run_optim" if args.use_optim else "./run"

# 遍历每个模型文件
for model in model_files:
    print(f"Running model: {model} with {'optimization' if args.use_optim else 'no optimization'}")
    tok_s_list = []
    
    # 多次运行模型
    for i in range(num_runs):
        print(f"Run {i + 1} for {model}...")
        # 调用运行命令
        process = subprocess.run(
            [executable, model],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 提取 tok/s
        output = process.stderr
        match = re.search(r"achieved tok/s:\s*([\d.]+)", output)
        if match:
            tok_s = float(match.group(1))
            tok_s_list.append(tok_s)
            print(f"Run {i + 1}: {tok_s} tok/s")
        else:
            print(f"Run {i + 1}: Efficiency not found!")
    
    # 计算平均效率
    if tok_s_list:
        avg_tok_s = sum(tok_s_list) / len(tok_s_list)
        results[model] = avg_tok_s
        print(f"Average tok/s for {model}: {avg_tok_s}")
    else:
        results[model] = None
        print(f"No efficiency data for {model}")

# 打印最终结果
print("\nFinal Results:")
for model, avg_tok_s in results.items():
    if avg_tok_s:
        print(f"{model}: {avg_tok_s:.2f} tok/s (average)")
    else:
        print(f"{model}: No data collected")
