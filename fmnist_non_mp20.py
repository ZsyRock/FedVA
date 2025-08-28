# lf.py — Unified Runner: dataset / net / poisoning all-in-one switches
import sys
import os
from loguru import logger

# ========== 顶部开关：按需只改这里 ==========

# 1) 数据集/分布（需要和你生成的 pickle 目录一致）
#    选项: "CIFAR10_IID", "CIFAR10_NONIID", "FMNIST_IID", "FMNIST_NONIID"
PRESET = "FMNIST_NONIID"

# 2) 模型结构：ResNet18(3x32x32) 或 轻量 CNN
#    选项: "resnet18" | "cnn"
NET_NAME = "cnn"

# 3) 投毒方式：无投毒 / 标签翻转 / 模型投毒（sign flip）
#    选项: "none" | "label_flip" | "model_sign"
POISONING = "model_sign"

# 4) 投毒相关参数
NUM_POISONED_WORKERS = 20        # 恶意客户端数（例如 20/100 = 20%）
SIGN_SCALE = 1.0                 # 仅 model_sign 生效：0.5/1/2 等
# 标签翻转方法：按需替换（比如 0->9 或 0->9 & 1->3）
from federated_learning.utils import replace_0_with_9
# from federated_learning.utils import replace_0_with_9_1_with_3
REPLACEMENT_METHOD = replace_0_with_9

# 5) 训练规模
NUM_WORKERS_PER_ROUND = 50       # 每轮抽取多少客户端参与
LOCAL_EPOCHS = 5                 # 每个客户端本地训练轮数
TOTAL_WORKERS = 100              # 全部客户端数量（需和生成数据时一致）

# 6) 其他常用训练参数
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
LR = 0.01
CUDA = True

# 7) 防御方法
DEFENCE = "PCA"                  # 或 "MI"；不想防御可写 None

# 8) 指标统计（0->9 作为默认目标映射，用于 SRC/ASR/TargetRecall 计算）
NUM_CLASSES = 10
SOURCE_CLASS = 0
TARGET_CLASS = 9

# ========== 下面通常无需改动 ==========

logger.remove()
logger.add(sys.stdout, level="INFO")

from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.nets.resnet_cifar import ResNet18
from federated_learning.nets.cifar_10_cnn import Cifar10CNN
from server import run_exp

# 数据集预设：同时指定 dataset 名和 data_loader_subdir（决定从哪个 pickle 目录加载）
PRESETS = {
    "CIFAR10_IID": {
        "dataset": "cifar10",
        "data_loader_subdir": "cifar10",
        "data_distribution_strategy": "iid",
    },
    "CIFAR10_NONIID": {
        "dataset": "cifar10",
        "data_loader_subdir": "cifar10_noniid",
        "data_distribution_strategy": "noniid",
    },
    "FMNIST_IID": {
        "dataset": "fashion-mnist",
        "data_loader_subdir": "fashion-mnist",
        "data_distribution_strategy": "iid",
    },
    "FMNIST_NONIID": {
        "dataset": "fashion-mnist",
        "data_loader_subdir": "fashion-mnist_noniid",
        "data_distribution_strategy": "noniid",
    },
}

# 自动查找下一个未被占用的实验编号
def find_next_exp_idx(base=2200):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base

def _pick_net_and_layer(net_name: str):
    """根据 NET_NAME 选择网络与被分析层名称"""
    if net_name.lower() == "resnet18":
        return ResNet18, "fc.weight"
    elif net_name.lower() == "cnn":
        return Cifar10CNN, "fc2.weight"
    else:
        raise ValueError(f"Unknown NET_NAME: {net_name}")

def _apply_poisoning_switch(args):
    """
    根据 POISONING 设置相关开关：
      - none:        关闭数据与模型投毒
      - label_flip:  开数据投毒，关闭模型投毒
      - model_sign:  关数据投毒，开 sign flip
    """
    if POISONING == "none":
        args.model_poison = None
        args.data_poison = False
    elif POISONING == "label_flip":
        args.model_poison = None
        args.data_poison = True
    elif POISONING == "model_sign":
        args.model_poison = "sign"
        args.sign_scale = SIGN_SCALE
        args.data_poison = False
    else:
        raise ValueError(f"Unknown POISONING: {POISONING}")
    # 恶意客户端的数据注入策略（用于 label_flip 或混合逻辑）
    args.mal_strat = "concat"

def run_model_poison_exp(replacement_method, num_poisoned_workers, kwargs, strategy, experiment_id):
    preset = PRESETS[PRESET]
    NetClass, layer_name = _pick_net_and_layer(NET_NAME)

    def config_modifier(args):
        # ===== 训练基本参数 =====
        args.batch_size = BATCH_SIZE
        args.test_batch_size = TEST_BATCH_SIZE
        args.lr = LR
        args.cuda = CUDA

        # ===== 数据集与加载路径（硬编码到生成的 pickle 目录）=====
        args.dataset = preset["dataset"]
        args.data_loader_subdir = preset["data_loader_subdir"]
        args.data_distribution_strategy = preset["data_distribution_strategy"]

        # ===== 模型与被分析层 =====
        args.net = NetClass
        args.layer_name = layer_name

        # ===== 投毒相关 =====
        _apply_poisoning_switch(args)

        # ===== 防御 =====
        args.defence = DEFENCE

        # ===== 联邦设置 =====
        args.num_workers = TOTAL_WORKERS
        args.local_epochs = LOCAL_EPOCHS

        # ===== 指标统计设置（SRC/ASR/TargetRecall 用）=====
        args.num_classes = NUM_CLASSES
        args.source_class = SOURCE_CLASS
        args.target_class = TARGET_CLASS

        return args

    run_exp(
        replacement_method=replacement_method,
        num_poisoned_workers=num_poisoned_workers,
        KWARGS=kwargs,  # 要包含 "NUM_WORKERS_PER_ROUND"
        client_selection_strategy=strategy,
        idx=experiment_id,
        config_modifier=config_modifier
    )

# ========= 主入口 =========
if __name__ == '__main__':
    START_EXP_IDX = find_next_exp_idx()
    NUM_EXP = 3

    KWARGS = {
        "NUM_WORKERS_PER_ROUND": NUM_WORKERS_PER_ROUND
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(
            REPLACEMENT_METHOD,
            NUM_POISONED_WORKERS,
            KWARGS,
            RandomSelectionStrategy(),
            experiment_id
        )
