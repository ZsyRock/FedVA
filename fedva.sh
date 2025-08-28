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




echo "==== fashion mnist noniid lf0 ===="
python fmnist_non_lf.py

echo "==== fashion mnist noniid lf10 ===="
python fmnist_non_lf10.py

echo "==== fashion mnist noniid lf20 ===="
python fmnist_non_lf20.py

echo "==== fashion mnist noniid lf30 ===="
python fmnist_non_lf30.py

echo "==== fashion mnist noniid lf40 ===="
python fmnist_non_lf40.py

echo "==== fashion mnist noniid mp0 ===="
python fmnist_non_mp.py

echo "==== fashion mnist noniid mp10 ===="
python fmnist_non_mp10.py

echo "==== fashion mnist noniid mp20 ===="
python fmnist_non_mp20.py

echo "==== fashion mnist noniid mp30 ===="
python fmnist_non_mp30.py

echo "==== fashion mnist noniid mp40 ===="
python fmnist_non_mp40.py

echo "==== cifar10 iid mp0 ===="
python cifar10_iid_mp0.py

echo "==== cifar10 iid mp40 ===="
python cifar10_iid_mp40.py

echo "==== cifar10 noniid lf20 ===="
python cifar10_non_lf20.py



echo "==== 所有实验运行完成 ===="
