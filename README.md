https://github.com/git-disl/DataPoisoning_FL

1. git clone https://github.com/ZsyRock/research-reproduction.git下载仓库后，cd ~/FedVA/，然后运行conda env create -f environment.yml即可创建环境

  - python=3.9
  - numpy=1.21
  - scikit-learn=0.24
  - matplotlib=3.5
  - pillow
  - scipy
  - pip:
      - charset-normalizer==3.4.1
      - cycler==0.11.0
      - fonttools==4.38.0
      - idna==3.10
      - joblib==1.3.2
      - kiwisolver==1.4.5
      - loguru==0.3.2
      - packaging==24.0
      - pyparsing==3.1.4
      - python-dateutil==2.9.0.post0
      - requests==2.31.0
      - six==1.17.0
      - torch==1.13.1
      - torchvision==0.14.1
      - typing-extensions==4.7.1
      - urllib3==2.0.7

2. 使用conda activate fedva激活环境fedva后:
    - 运行python generate_data_distribution.py,
    - 运行python generate_default_models.py,
    
    得到：

    ├── data_loaders
    │   ├── cifar10
    │   │   ├── test_data_loader.pickle
    │   │   └── train_data_loader.pickle
    │   └── fashion-mnist
    │       ├── test_data_loader.pickle
    │       └── train_data_loader.pickle
    ├── default_models
    │   ├── Cifar10CNN.model
    │   └── FashionMNISTCNN.model

    - 这两个文件是默认的训练数据和模型参数，后续的训练和测试都需要用到。

3. 在lf.py文件中修改投毒数量，然后运行python lf.py可以在根目录下生成
    - /figures/
    - /logs/
    - /tabular/


通过HPC做实验：

遇到ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
运行命令find $CONDA_PREFIX -name "libstdc++.so.6"
运行命令export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
再运行python generate_data_distribution.py

创建脚本fedva.sh，并且复制下面的代码到fedva.sh文件中，再通过sbatch fedva.sh来开始实验：
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
python lf.py

