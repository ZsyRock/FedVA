import sys
import os
from loguru import logger

# ==== 攻击映射（按需切换）====
from federated_learning.utils import replace_0_with_9
REPLACEMENT_METHOD = replace_0_with_9

from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.nets.resnet_cifar import ResNet18
# from federated_learning.nets.cifar_10_cnn import Cifar10CNN  # 若想用CNN再切换
from server import run_exp

logger.remove()
logger.add(sys.stdout, level="INFO")


# ========= [一键切换] 数据集/分布预设 =========
# 选项: "CIFAR10_IID", "CIFAR10_NONIID", "FMNIST_IID", "FMNIST_NONIID"
PRESET = "FMNIST_NONIID"

PRESETS = {
    # 备注：子目录需与生成的 pickle 路径一致
    "CIFAR10_IID": {
        "dataset": "cifar10",
        "data_loader_subdir": "cifar10",
        "data_distribution_strategy": "iid",     # 仅用于日志语义；实际以 data_loader_subdir 加载
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
def find_next_exp_idx(base=3000):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base


def run_model_poison_exp(replacement_method, num_poisoned_workers, kwargs, strategy, experiment_id):
    preset = PRESETS[PRESET]

    def config_modifier(args):
        # ===== 基本训练超参（可按需要微调）=====
        args.batch_size = 128
        args.test_batch_size = 1000
        args.lr = 0.01
        args.cuda = True

        # ===== 关键：切换数据集/分布/数据目录 =====
        # 这里 *硬编码* 告诉加载器去哪儿找 pickle
        args.dataset = preset["dataset"]
        args.data_loader_subdir = preset["data_loader_subdir"]
        args.data_distribution_strategy = preset["data_distribution_strategy"]

        # ===== 模型结构（ResNet18 适配 3×32×32）=====
        args.net = ResNet18
        args.layer_name = "fc.weight"   # 若切到 Cifar10CNN，请改成 "fc2.weight"

        # ===== 攻击/防御设置 =====
        args.model_poison = None          # 或 "sign"
        args.data_poison = True           # 标签翻转
        args.mal_strat = "concat"
        args.defence = "PCA"

        # ===== 联邦设置 =====
        args.num_workers = 100
        # Dirichlet α 在生成 pickle 时已固定，这里仅作记号：
        # args.noniid_alpha = 0.5

        # ===== 指标统计相关（0→9 场景）=====
        args.num_classes = 10
        args.source_class = 0
        args.target_class = 9

        # ===== 本地多 epoch =====
        args.local_epochs = 5

        return args

    run_exp(
        replacement_method=replacement_method,
        num_poisoned_workers=num_poisoned_workers,
        KWARGS=kwargs,
        client_selection_strategy=strategy,
        idx=experiment_id,
        config_modifier=config_modifier
    )


# ========= 主入口 =========
if __name__ == '__main__':
    START_EXP_IDX = find_next_exp_idx()
    NUM_EXP = 1

    # 想做 baseline 就把投毒设为 0
    NUM_POISONED_WORKERS = 0

    KWARGS = {
        # 抽取比例按你需要改（例如 50/100 客户端）
        "NUM_WORKERS_PER_ROUND": 50
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
