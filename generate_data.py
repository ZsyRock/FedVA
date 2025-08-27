# generate_all_pickles.py
# 生成四套可直接用于 ResNet18 的 DataLoader pickle：
# 1) CIFAR10 (IID)                -> data_loaders/cifar10/
# 2) CIFAR10 (Dirichlet non-IID)  -> data_loaders/cifar10_noniid/
# 3) Fashion-MNIST (IID, 3x32x32) -> data_loaders/fashion-mnist/
# 4) Fashion-MNIST (non-IID, 3x32x32) -> data_loaders/fashion-mnist_noniid/

from loguru import logger
import os
import pathlib
import pickle
import numpy as np
import torch

# torchvision 仅用于把 Fashion-MNIST 变成 3x32x32
from torchvision import datasets, transforms

# 项目里的封装
from federated_learning.arguments import Arguments
from federated_learning.datasets import CIFAR10Dataset, FashionMNISTDataset
from federated_learning.datasets.dataset import Dataset
from federated_learning.utils.data_loader_utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution.noniid_dirichlet import distribute_noniid_dirichlet

# -------------------------
# 小工具
# -------------------------
def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def save_pkl(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved: {path}")

def label_hist_from_tuples(data_tuples):
    """data_tuples: list of (tensor, label_tensor)"""
    labels = [y.item() for _, y in data_tuples]
    uniq, cnt = np.unique(labels, return_counts=True)
    return dict(zip(uniq, cnt))

# -------------------------
# 1) CIFAR10 - IID
# -------------------------
def build_cifar10_iid(args: Arguments):
    logger.info("=== Building CIFAR10 (IID) ===")
    dataset = CIFAR10Dataset(args)
    train = Dataset.get_data_loader_from_data(
        batch_size=args.get_batch_size(),
        X=dataset.get_train_dataset()[0],
        Y=dataset.get_train_dataset()[1],
        shuffle=True
    )
    test = Dataset.get_data_loader_from_data(
        batch_size=args.get_test_batch_size(),
        X=dataset.get_test_dataset()[0],
        Y=dataset.get_test_dataset()[1],
        shuffle=False
    )
    save_pkl(train, "data_loaders/cifar10/train_data_loader.pickle")
    save_pkl(test,  "data_loaders/cifar10/test_data_loader.pickle")

# -------------------------
# 2) CIFAR10 - non-IID (Dirichlet)
# -------------------------
def build_cifar10_noniid(args: Arguments, alpha: float):
    logger.info(f"=== Building CIFAR10 (non-IID, alpha={alpha}) ===")
    dataset = CIFAR10Dataset(args)

    full_train = Dataset.get_data_loader_from_data(
        batch_size=args.get_batch_size(),
        X=dataset.get_train_dataset()[0],
        Y=dataset.get_train_dataset()[1],
        shuffle=True
    )

    distributed = distribute_noniid_dirichlet(
        full_train,
        num_workers=args.get_num_workers(),
        alpha=alpha
    )

    # 打印每 client 标签分布（可选）
    for cid, tuples in enumerate(distributed):
        logger.info(f"[CIFAR10 non-IID] Client #{cid} label dist: {label_hist_from_tuples(tuples)}")

    loaders = generate_data_loaders_from_distributed_dataset(distributed, args.get_batch_size())

    test = Dataset.get_data_loader_from_data(
        batch_size=args.get_test_batch_size(),
        X=dataset.get_test_dataset()[0],
        Y=dataset.get_test_dataset()[1],
        shuffle=False
    )

    save_pkl(loaders, "data_loaders/cifar10_noniid/train_data_loader.pickle")
    save_pkl(test,   "data_loaders/cifar10_noniid/test_data_loader.pickle")

# -------------------------
# 3) Fashion-MNIST - IID (转为 3x32x32)
# -------------------------
def build_fmnist_iid(args: Arguments):
    logger.info("=== Building Fashion-MNIST (IID, 3x32x32) ===")

    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    train_set = datasets.FashionMNIST(root="data", train=True,  download=True, transform=tfm)
    test_set  = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)

    # 转为 (X,Y) 张量形式
    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))]).numpy()
    Y_train = np.array([train_set[i][1] for i in range(len(train_set))])
    X_test  = torch.stack([test_set[i][0]  for i in range(len(test_set))]).numpy()
    Y_test  = np.array([test_set[i][1] for i in range(len(test_set))])


    train = Dataset.get_data_loader_from_data(args.get_batch_size(), X_train, Y_train, shuffle=True)
    test  = Dataset.get_data_loader_from_data(args.get_test_batch_size(), X_test, Y_test, shuffle=False)

    save_pkl(train, "data_loaders/fashion-mnist/train_data_loader.pickle")
    save_pkl(test,  "data_loaders/fashion-mnist/test_data_loader.pickle")

# -------------------------
# 4) Fashion-MNIST - non-IID (Dirichlet, 转为 3x32x32)
# -------------------------
def build_fmnist_noniid(args: Arguments, alpha: float):
    logger.info(f"=== Building Fashion-MNIST (non-IID, 3x32x32, alpha={alpha}) ===")

    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    train_set = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)
    # 转为 (X,Y)
    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))]).numpy()
    Y_train = np.array([train_set[i][1] for i in range(len(train_set))])


    full_train = Dataset.get_data_loader_from_data(
        batch_size=args.get_batch_size(),
        X=X_train, Y=Y_train, shuffle=True
    )

    distributed = distribute_noniid_dirichlet(
        full_train,
        num_workers=args.get_num_workers(),
        alpha=alpha
    )

    for cid, tuples in enumerate(distributed):
        logger.info(f"[Fashion-MNIST non-IID] Client #{cid} label dist: {label_hist_from_tuples(tuples)}")

    loaders = generate_data_loaders_from_distributed_dataset(distributed, args.get_batch_size())

    # Test 同样走 3x32x32
    test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)
    X_test  = torch.stack([test_set[i][0] for i in range(len(test_set))]).numpy()
    Y_test  = np.array([test_set[i][1] for i in range(len(test_set))])
    test    = Dataset.get_data_loader_from_data(args.get_test_batch_size(), X_test, Y_test, shuffle=False)


    save_pkl(loaders, "data_loaders/fashion-mnist_noniid/train_data_loader.pickle")
    save_pkl(test,    "data_loaders/fashion-mnist_noniid/test_data_loader.pickle")

# -------------------------
# 主入口
# -------------------------
if __name__ == "__main__":
    args = Arguments(logger)

    # 可以在这里统一调 batch / test_batch / num_workers / alpha
    # （默认会从 Arguments 里取 get_batch_size/get_test_batch_size/get_num_workers）
    alpha_cifar = 0.5
    alpha_fmnist = 0.5

    # 1) CIFAR10 IID
    build_cifar10_iid(args)

    # 2) CIFAR10 non-IID
    build_cifar10_noniid(args, alpha=alpha_cifar)

    # 3) Fashion-MNIST IID (3x32x32)
    build_fmnist_iid(args)

    # 4) Fashion-MNIST non-IID (3x32x32)
    build_fmnist_noniid(args, alpha=alpha_fmnist)

    logger.success("All four pickle sets generated successfully.")
