import sys
import os
from loguru import logger

from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.nets.resnet_cifar import ResNet18
# 如需改用 CNN：from federated_learning.nets.cifar_10_cnn import Cifar10CNN
from server import run_exp

logger.remove()
logger.add(sys.stdout, level="INFO")


def find_next_exp_idx(base=4000):
    """避免覆盖现有日志/模型文件，自动选择下一个可用实验编号。lf.py 3000 开始，mp.py 4000 开始"""
    while os.path.exists(f"logs/{base}.log") or os.path.exists(f"{base}_models"):
        base += 1
    return base


def run_model_poison_exp(num_poisoned_workers, kwargs, strategy, experiment_id):
    def config_modifier(args):
        # ===== 训练基本参数 =====
        args.batch_size = 64            # 模型投毒一般不受small batch限制，可比 lf.py 稍大
        args.test_batch_size = 500
        args.lr = 0.01
        args.cuda = True

        # ===== 模型与关键层 =====
        args.net = ResNet18
        args.layer_name = "fc.weight"   # ResNet18 的最后一层
        # 如果切到 CNN：
        # args.net = Cifar10CNN
        # args.layer_name = "fc2.weight"

        # ===== 攻击：模型投毒（sign flipping）=====
        args.model_poison = "sign"      # 触发 client.sign_attack()
        args.sign_scale = 1.0           # 攻击强度：>0；1.0=直接取反；2.0/3.0更猛
        args.data_poison = False        # 关闭数据投毒
        args.mal_strat = None           # 无需数据投毒策略

        # ===== 防御：使用 MI（作者核心点）=====
        args.defence = "MI"             # 你也可换成 "PCA" 试对比

        # ===== 数据分布 =====
        args.num_workers = kwargs.get("TOTAL_NUM_WORKERS", 100)
        args.data_distribution_strategy = "noniid"
        args.noniid_alpha = kwargs.get("NONIID_ALPHA", 1.0)

        # ===== MI 的分桶参数（不写也有默认：fed=0.2/grey=0.5）=====
        args.fed_pct = kwargs.get("MI_FED_PCT", 0.2)
        args.grey_pct = kwargs.get("MI_GREY_PCT", 0.5)

        # ===== 评估用 =====
        args.num_classes = 10
        args.source_class = 0
        args.target_class = 9

        return args

    run_exp(
        replacement_method=None,             # 模型投毒无需标签替换函数
        num_poisoned_workers=num_poisoned_workers,
        KWARGS=kwargs,                       # 里边要包含 NUM_WORKERS_PER_ROUND
        client_selection_strategy=strategy,  # 随机选客户端
        idx=experiment_id,
        config_modifier=config_modifier
    )


if __name__ == "__main__":
    START_EXP_IDX = find_next_exp_idx()
    NUM_EXP = 1

    # ===== 关键实验开关 =====
    NUM_POISONED_WORKERS = 30                 # 恶意客户端个数
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 100,         # 每轮参与客户端数（随机）
        "TOTAL_NUM_WORKERS": 100,             # 总客户端数（用于上面 args.num_workers）
        "NONIID_ALPHA": 1.0,                  # Data 分布参数（Noniid）
        # MI 的分桶（可省略，用 arguments.py 默认 0.2 / 0.5）
        "MI_FED_PCT": 0.2,
        "MI_GREY_PCT": 0.5,
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_model_poison_exp(NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
