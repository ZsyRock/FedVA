#!/bin/bash

#SBATCH -p a100                    # 使用 A100 分区
#SBATCH --gres=gpu:2              # 申请 2 张 GPU
#SBATCH --nodes=1                 # 使用 1 个计算节点
#SBATCH --time=60:00:00           # 最长运行时间为 60 小时
#SBATCH --job-name=fedva_train    # 作业名称
#SBATCH --output=fedva_output.log # 输出日志文件
#SBATCH --mem=80G                 # 申请 80 GB 内存

set -e  # 脚本出错时立即退出

# 加载环境
source ~/.bashrc
conda activate fedva

# 切换到工作目录
cd /iridisfs/home/sz1c24/FedVA

# 设置 LD_LIBRARY_PATH，避免 libstdc++ 报错
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 启动训练
# ====== 依次运行多个实验 ======


echo "==== 开始运行 mp40.py ===="
python mp40.py

echo "==== 开始运行 mp30.py ===="
python mp30.py

echo "==== 开始运行 mp20.py ===="
python mp20.py

echo "==== 开始运行 mp10.py ===="
python mp10.py


echo "==== 所有实验运行完成 ===="
