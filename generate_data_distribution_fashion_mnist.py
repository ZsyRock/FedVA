from loguru import logger
import os, pathlib, pickle, numpy as np

from federated_learning.arguments import Arguments
from federated_learning.datasets import FashionMNISTDataset   # 如果项目里叫别名，按你仓库改
from federated_learning.datasets.data_distribution.noniid_dirichlet import distribute_noniid_dirichlet
from federated_learning.datasets.dataset import Dataset
from federated_learning.utils.data_loader_utils import generate_data_loaders_from_distributed_dataset

def save(obj, path):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved to: {path}")

if __name__ == "__main__":
    args = Arguments(logger)
    # 关键：指向 fashion-mnist
    args.dataset = "fashion-mnist"          # 若类里用 get_dataset_name，也可 args.set_dataset("fashion-mnist")
    args.data_distribution_strategy = "noniid"
    args.noniid_alpha = 1.0                 # 需要更偏/更混合就调 0.5 / 0.3 / 2.0 等

    # 为了兼容你现有的 ResNet18(3×32×32)，把灰度图重复成 3 通道并 resize 到 32
    args.force_grayscale_to_rgb = True      # 如果项目没有这个字段，不用管，这里只是个标记说明
    args.resize_to_32 = True

    dataset = FashionMNISTDataset(args)

    train_set = dataset.get_train_dataset()
    train_loader_full = Dataset.get_data_loader_from_data(
        batch_size=args.get_batch_size(), X=train_set[0], Y=train_set[1], shuffle=True
    )

    distributed = distribute_noniid_dirichlet(
        train_loader_full,
        num_workers=args.get_num_workers(),
        alpha=args.noniid_alpha
    )

    #（可选）打印每个 client 的标签分布
    for cid, data in enumerate(distributed):
        labels = [lab.item() for _, lab in data]
        uniq, cnt = np.unique(labels, return_counts=True)
        logger.info(f"[Client #{cid}] label dist: {dict(zip(uniq, cnt))}")

    train_loaders = generate_data_loaders_from_distributed_dataset(distributed, args.get_batch_size())
    test_loader = dataset.get_test_loader(args.get_test_batch_size())

    save_dir = "data_loaders/fashion-mnist_noniid"
    save(train_loaders, os.path.join(save_dir, "train_data_loader.pickle"))
    save(test_loader,  os.path.join(save_dir, "test_data_loader.pickle"))

    logger.success("Fashion-MNIST non-IID pickles generated.")
