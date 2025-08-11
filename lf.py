import sys
import os

#from federated_learning.utils import replace_0_with_9_1_with_3
from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.nets.resnet_cifar import ResNet18
from federated_learning.nets.cifar_10_cnn import Cifar10CNN 
from server import run_exp
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")

# 评估 0→9
from federated_learning.utils import replace_0_with_9
REPLACEMENT_METHOD = replace_0_with_9

# 自动查找下一个未被占用的实验编号
def find_next_exp_idx(base=3000):
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base

# 设置实验运行逻辑（启用标签翻转攻击 + Non-IID 分布）
def run_model_poison_exp(replacement_method, num_poisoned_workers, kwargs, strategy, experiment_id):
    def config_modifier(args):
        args.batch_size = 8
        args.test_batch_size = 500
        args.lr = 0.01
        args.cuda = True
        # === 切换到 ResNet18（关键两行） ===
        args.net = ResNet18 #可以改为Cifar10CNN，启用CNN
        args.layer_name = "fc.weight"   # ResNet18 最后一层权重名 "fc.weight"，如果使用CNN就替换为"fc2.weight"
        # —— 其他训练/攻击/防御设置——
        args.model_poison = None
        args.data_poison = True
        args.mal_strat = "concat"
        args.defence = "PCA"
        args.num_workers = 100
        args.data_distribution_strategy = "noniid"
        args.noniid_alpha = 1.0
        # 标签翻转统计：0→9
        args.num_classes = 10
        args.source_class = 0
        args.target_class = 9

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
    NUM_POISONED_WORKERS = 30
    # 这里不要再重复定义 REPLACEMENT_METHOD
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 100
    }
    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)