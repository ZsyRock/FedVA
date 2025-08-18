# mp.py — Model Poisoning (sign-flip) runner
import sys
import os
from federated_learning.utils import replace_0_with_9   # 保留占位，不用也无妨
from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.nets.resnet_cifar import ResNet18
from federated_learning.nets.cifar_10_cnn import Cifar10CNN
from server import run_exp
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

REPLACEMENT_METHOD = replace_0_with_9  # data_poison=False 时不会用到

# 自动查找下一个未被占用的实验编号（从 4000 开始）
def find_next_exp_idx(base=4400):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base

def run_model_poison_exp(replacement_method, num_poisoned_workers, kwargs, strategy, experiment_id):
    def config_modifier(args):
        # ===== 训练超参（与你的 lf.py 基本一致）=====
        args.batch_size = 128
        args.test_batch_size = 1000
        args.lr = 0.01
        args.cuda = True

        # ===== 模型与被分析层 =====
        args.net = ResNet18                 # 如需 CNN：args.net = Cifar10CNN
        args.layer_name = "fc.weight"       # 若用 Cifar10CNN 则改为 "fc2.weight"

        # ===== 关键：启用“模型投毒（sign-flip）”，关闭数据投毒 =====
        args.model_poison = "sign"          # 触发 client.sign_attack(...)
        args.sign_scale = 0.5               # 强度可做 0.5/1/2 消融
        args.data_poison = False            # 只做模型投毒，避免与标签翻转混合

        # ===== 其它保持不变 =====
        args.mal_strat = "concat"
        args.defence = "PCA"                # 或 "MI"
        args.num_workers = 100
        args.data_distribution_strategy = "noniid"
        args.num_classes = 10
        args.source_class = 0
        args.target_class = 9
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
    START_EXP_IDX = find_next_exp_idx()     # 从 4000 起，自动递增
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 40               # 恶意客户端数量（可按需改）
    KWARGS = { "NUM_WORKERS_PER_ROUND": 50 }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(
            REPLACEMENT_METHOD,
            NUM_POISONED_WORKERS,
            KWARGS,
            RandomSelectionStrategy(),
            experiment_id
        )
