# ====== 基础导入 ======
import os
import copy
import torch
import numpy as np
from loguru import logger
from sklearn.feature_selection import mutual_info_regression
import time
import sys
import pickle
import csv

# ====== 项目内部模块导入 ======
from client import Client
from federated_learning.arguments import Arguments
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import (
    generate_data_loaders_from_distributed_dataset,
    average_nn_parameters,
    convert_distributed_data_into_numpy,
    identify_random_elements,
    save_results,
    load_train_data_loader,
    load_test_data_loader,
    generate_experiment_ids,
    convert_results_to_csv,
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from defense import (
    load_models, cluster_step1_GM, plot_gradients_2d,
    two_cluster_GM, one_cluster
)

# ====== 全局参数 ======
CLASS_NUM = 1
#LAYER_NAME = "fc2.weight"
THRESHOLD = 0.8
DISCARD_THD = 20

# ===== Part 1: Flatten 工具函数 =====
def flatten_layers(model_param):
    p = []
    for layer in model_param:
        param = model_param[layer]
        if isinstance(param, torch.Tensor):
            p.append(param.cpu().numpy().flatten())
        else:
            p.append(param.flatten())
    return np.concatenate(p, axis=0)

def _cleanup_old_models(models_dir, keep_epoch, keep_last_k=1):
    """
    删除早于 keep_epoch-(keep_last_k-1) 的所有模型快照文件。
    文件名格式假定为: model_{client}_{epoch}_{suffix}.model
    """
    if not os.path.isdir(models_dir):
        return
    min_keep_epoch = max(1, keep_epoch - keep_last_k + 1)
    for fname in os.listdir(models_dir):
        if not fname.startswith("model_") or not fname.endswith(".model"):
            continue
        parts = fname[:-6].split("_")  # 去掉 .model
        if len(parts) < 4:
            continue
        try:
            # model_{client}_{epoch}_{suffix}
            ep = int(parts[2])
        except ValueError:
            continue
        if ep < min_keep_epoch:
            try:
                os.remove(os.path.join(models_dir, fname))
            except Exception as e:
                logger.warning(f"[CLEANUP] Failed to remove {fname}: {e}")

# ===== Part 2: 主调度入口函数 =====
def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, config_modifier=None):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(
        idx, 1)
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    if config_modifier is not None:
        args = config_modifier(args)

    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    global DISCARD_THD
    DISCARD_THD = int(args.get_num_epochs() * 0.25)

    if args.get_data_distribution_strategy() == "noniid":
        distributed_train_dataset = train_data_loader  # 已经是 list[DataLoader]
    else:
        from federated_learning.datasets.data_distribution.iid_equal import distribute_batches_equally
        distributed_train_dataset = distribute_batches_equally(
            train_data_loader,
            args.get_num_workers()
        ) ## noniid 分布时不需要转换
    distributed_train_dataset = convert_distributed_data_into_numpy(
        distributed_train_dataset)

    poisoned_workers = identify_random_elements(
        args.get_num_workers(), args.get_num_poisoned_workers())

    if args.get_data_distribution_strategy() == "noniid":
        train_data_loaders = train_data_loader  # 已是 list of DataLoader
    else:
        from federated_learning.datasets.data_distribution.iid_equal import distribute_batches_equally
        train_data_loaders = distribute_batches_equally(
            train_data_loader,
            args.get_num_workers()
        ) ## noniid 分布时不需要转换

    if args.get_data_distribution_strategy() == "noniid":
        for client_id, loader in enumerate(train_data_loaders):
            label_counter = {}
            for _, target in loader:
                for label in target.numpy():
                    label_counter[label] = label_counter.get(label, 0) + 1
            logger.info(f"[Run-time Check] Client #{client_id} label distribution: {label_counter}") ## logging label分布

    clients = create_clients(args, train_data_loaders, test_data_loader)
    for id in range(len(clients)):
        if clients[id].client_idx in poisoned_workers:
            clients[id].mal = True
            if args.data_poison:
                clients[id].poison_data(replacement_method)

    results, worker_selection = run_machine_learning(
        clients, args, poisoned_workers)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)

# ===== Part 3: 创建客户端对象 =====
def create_clients(args, train_data_loaders, test_data_loader):
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))
    return clients

# ===== Part 4: 联邦训练主循环 =====
def run_machine_learning(clients, args, poisoned_workers):
    epoch_results = []
    worker_selection = []
    mal_count = [0] * len(clients)

    for epoch in range(1, args.get_num_epochs() + 1):
        results_dict, workers_selected, mal_count = train_subset_of_clients(
            epoch, args, clients, mal_count, poisoned_workers
        )
        epoch_results.append(results_dict)
        worker_selection.append(workers_selected)

    args.get_logger().info(f"Final malicious count: {mal_count}, discard threshold: {DISCARD_THD}")
    args.get_logger().info("Server validation data required: False")  # FedVA 的卖点

    # === 写 CSV（每轮一行）===
    save_dir = args.get_save_model_folder_path()
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics_per_epoch.csv")
    fieldnames = [
        "epoch", "global_acc", "target_recall", "src", "asr",
        "fp", "fn", "fp_rate", "fn_rate",
        "benign_killed", "comm_cost_bytes", "epoch_time_sec"
    ]
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_results:
                # 缺失字段填充
                safe_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(safe_row)
        args.get_logger().info(f"Per-epoch metrics saved to: {csv_path}")
    except Exception as e:
        args.get_logger().warning(f"Failed to write CSV: {e}")

    # === 也保存 JSON（便于机器读）===
    try:
        save_results(epoch_results, os.path.join(save_dir, "metrics_per_epoch.json"))
    except Exception as e:
        args.get_logger().warning(f"Failed to save JSON metrics: {e}")

    # === 最终总结：最后一轮 + 最佳一轮 ===
    final = epoch_results[-1]
    best = max(epoch_results, key=lambda x: x.get("global_acc", float("-inf")))

    args.get_logger().info(
        "[Final Summary] "
        f"Last Acc={final.get('global_acc', float('nan')):.2f}%, "
        f"Last TargetRecall={final.get('target_recall', float('nan')):.4f}, "
        f"Last SRC={final.get('src', float('nan')):.4f}, "
        f"Last ASR={final.get('asr', float('nan')):.4f}, "
        f"Last BenignKilled={final.get('benign_killed', 0)}"
    )
    args.get_logger().info(
        "[Best Summary]  "
        f"BestEpoch={best.get('epoch', -1)} | "
        f"Acc={best.get('global_acc', float('nan')):.2f}% | "
        f"TargetRecall={best.get('target_recall', float('nan')):.4f} | "
        f"SRC={best.get('src', float('nan')):.4f} | "
        f"ASR={best.get('asr', float('nan')):.4f} | "
        f"BenignKilled={best.get('benign_killed', 0)}"
    )

    return epoch_results, worker_selection


# ===== Part 5: 每轮客户端训练和防御判断 =====
def train_subset_of_clients(epoch, args, clients, mal_count, poisoned_workers):
    logger.info(f"[DEBUG] Using strategy: {args.get_round_worker_selection_strategy().__class__.__name__}")
    logger.info(f"[DEBUG] workers length before selection: {len(list(range(args.get_num_workers())))}")
    logger.info(f"[DEBUG] poisoned_workers: {poisoned_workers}")

    epoch_t0 = time.time()  # NEW: 计时

    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())), poisoned_workers, kwargs)
    logger.info(f"[DEBUG] random_workers this round (len={len(random_workers)}): {random_workers}")
    logger.info(f"[DEBUG] mal_count before training: {mal_count}")

    layer_name = args.layer_name
    old_layer_params = copy.deepcopy(
        list(get_layer_parameters(clients[0].get_nn_parameters(), layer_name)[CLASS_NUM]))

    # === 本地训练（原逻辑不变）===
    for client_idx in random_workers:
        if mal_count[client_idx] > DISCARD_THD:
            continue
        if clients[client_idx].mal:
            if args.data_poison:
                if args.mal_strat == 'concat':
                    clients[client_idx].concat_train(epoch)
                else:
                    clients[client_idx].blend_train(epoch)
            else:
                clients[client_idx].blend_train(epoch)
            if args.model_poison == "sign":
                clients[client_idx].sign_attack(epoch)
        else:
            clients[client_idx].benign_train(epoch)

    # === 防御筛选（原逻辑不变，额外保留 discard_list）===
    if args.defence:
        exp_id = args.get_save_model_folder_path().split("_")[0]
        fig_save_dir = f"figures/GS_{exp_id}"
        os.makedirs(fig_save_dir, exist_ok=True)

        if args.defence == "PCA":
            benign_models, mal_models, grey_models, fr_models, gradiants, pca, scl = PCA_clustering_selection(args, epoch, fig_save_dir)
        elif args.defence == "MI":
            benign_models, mal_models, grey_models, fr_models = mutual_info_clustering_selection(args, epoch)
            gradiants = []
            pca = scl = None
        else:
            raise ValueError(f"Unknown defence method: {args.defence}")

        cls_check, acc_check = class_validation(clients, fr_models, grey_models)
        val_check = verify_by_fr(clients, fr_models, grey_models)

        for gr, cval, aval, vval in zip(grey_models, cls_check, acc_check, val_check):
            if not cval or not aval or not vval:
                mal_models.append(gr)
                mal_count[gr] += 1
            else:
                benign_models.append(gr)

        discard_list = []
        for i, count in enumerate(mal_count):
            if count > DISCARD_THD:
                if i in benign_models:
                    benign_models.remove(i)
                mal_models.append(i)
                discard_list.append(i)

        if gradiants:
            plot_gradients_2d(gradients=gradiants, marker_list=[benign_models], save_name=f"Updated_Epoch_{epoch}.jpg",
                              label=['benign', 'mal'], save_dir=fig_save_dir)

        parameters = [clients[i].get_nn_parameters() for i in benign_models]
    else:
        parameters = [clients[i].get_nn_parameters() for i in random_workers]
        discard_list = []  # NEW: 保证存在
        
    # 得到 parameters 之后,清理旧 epoch 的模型快照，只保留最新的文件；如果想保留最近 K 轮，把 keep_last_k 改成 K即可
    try:
        _cleanup_old_models(args.get_save_model_folder_path(), keep_epoch=epoch, keep_last_k=1)
    except Exception as e:
        logger.warning(f"[CLEANUP] Exception during model cleanup at epoch {epoch}: {e}")
    # === 通信开销（近似）：上传参数字节数 × 被采纳客户端数 ===
    try:
        model_bytes = sys.getsizeof(pickle.dumps(clients[0].get_nn_parameters()))
        comm_cost_bytes = model_bytes * len(parameters)
    except Exception:
        comm_cost_bytes = float('nan')

    # === 聚合 + 下发（原逻辑）===
    new_nn_params = average_nn_parameters(parameters)
    for client in clients:
        client.update_nn_parameters(new_nn_params)

    # === 全局测试（聚合后，用所有客户端的 test 结果汇总）===
    TARGET_CLASS = getattr(args, "target_class", 0)
    SOURCE_CLASS = getattr(args, "source_class", None)
    NUM_CLASSES = getattr(args, "num_classes", 10)

    all_acc = []
    cm_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for client in clients:
        acc, _, prec, recall, cm = client.test()
        all_acc.append(acc)
        # 汇总混淆矩阵（更稳定）
        if cm.shape == cm_sum.shape:
            cm_sum += cm

    # 从 cm_sum 计算全局 per-class recall
    row_sum = cm_sum.sum(axis=1).clip(min=1)
    per_class_recall = np.diag(cm_sum) / row_sum
    target_recall = float(per_class_recall[TARGET_CLASS]) if 0 <= TARGET_CLASS < NUM_CLASSES else float('nan')
    source_recall = float(per_class_recall[SOURCE_CLASS]) if 0 <= SOURCE_CLASS < NUM_CLASSES else float('nan')


    # ASR 定义（label-flip 场景）：源类被错到目标类的比例
    if isinstance(SOURCE_CLASS, int) and 0 <= SOURCE_CLASS < NUM_CLASSES and \
       isinstance(TARGET_CLASS, int) and 0 <= TARGET_CLASS < NUM_CLASSES:
        src_total = cm_sum[SOURCE_CLASS, :].sum()
        asr = float(cm_sum[SOURCE_CLASS, TARGET_CLASS] / src_total) if src_total > 0 else float('nan')
    else:
        asr = float('nan')  # backdoor 场景或未配置时，用 NaN

    global_acc = float(np.mean(all_acc)) if len(all_acc) else float('nan')

    # === 检测类指标：FP/FN 基于“防御判定 vs 真实恶意集合” ===
    fp = fn = 0
    fp_rate = fn_rate = float('nan')
    if args.defence:
        mal_set = set(mal_models)
        benign_set = set(benign_models)
        poison_set = set(poisoned_workers)

        fp = len(mal_set - poison_set)      # 良性被判恶意
        fn = len(benign_set & poison_set)   # 恶意被判良性

        num_benign = args.get_num_workers() - len(poisoned_workers)
        num_mal = len(poisoned_workers)
        fp_rate = (fp / num_benign) if num_benign > 0 else float('nan')
        fn_rate = (fn / num_mal) if num_mal > 0 else float('nan')

    benign_killed = len(set(discard_list) - set(poisoned_workers))
    epoch_time_sec = time.time() - epoch_t0

    # === 汇总当轮指标（用于 CSV/日志/最终总结）===
    results_dict = {
        "epoch": epoch,
        "global_acc": global_acc,
        "target_recall": target_recall,
        "src": source_recall,
        "asr": asr,
        "fp": fp,
        "fn": fn,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "benign_killed": benign_killed,
        "comm_cost_bytes": comm_cost_bytes,
        "epoch_time_sec": epoch_time_sec,
    }

    # 仍然保持原函数返回结构（第三个是 mal_count）
    return results_dict, random_workers, mal_count


# ===== Part 6: PCA聚类分析防御方法 =====
def PCA_clustering_selection(args, epoch, fig_save_dir):
    MODELS_PATH = args.get_save_model_folder_path()
    model_files = sorted(os.listdir(MODELS_PATH))
    param_diff, worker_ids = [], []

    start_file = get_model_files_for_suffix(
        get_model_files_for_epoch(model_files, epoch), 
        args.get_epoch_save_start_suffix()
    )[0]
    start_model = load_models(args, [os.path.join(MODELS_PATH, start_file)])[0]
    layer_name = args.layer_name
    start_layer = list(get_layer_parameters(start_model.get_nn_parameters(), layer_name)[CLASS_NUM])

    end_files = get_model_files_for_suffix(
        get_model_files_for_epoch(model_files, epoch), 
        args.get_epoch_save_end_suffix()
    )

    for f in end_files:
        worker_id = get_worker_num_from_model_file_name(f)
        end_model = load_models(args, [os.path.join(MODELS_PATH, f)])[0]
        end_layer = list(get_layer_parameters(end_model.get_nn_parameters(), layer_name)[CLASS_NUM])
        gradient = calculate_parameter_gradients(logger, start_layer, end_layer).flatten()
        param_diff.append(gradient)
        worker_ids.append(worker_id)

    # 安全兜底：如果这一轮没有任何模型差分，直接返回空集合，避免后续报错
    if len(param_diff) == 0:
        return [], [], [], [], [], None, None

    scaled_diff, scaler = apply_standard_scaler(param_diff)
    dim_reduced, pca = calculate_pca_of_gradients(logger, scaled_diff, 2)

    # 使用双簇 GMM（non-IID 更稳）
    benign, mal, grey, fr = two_cluster_GM(
        dim_reduced, worker_ids,
        fed_pct=0.2, grey_pct=0.2,
        title=f"GaussianMixture_2C_E{epoch}",
        save_dir=fig_save_dir
    )

    # 现在再打印，变量已就绪
    logger.info(f"[PCA@E{epoch}] benign={len(benign)}, grey={len(grey)}, bad={len(mal)}, fr={len(fr)}")

    return list(benign), list(mal), list(grey), list(fr), list(zip(worker_ids, dim_reduced)), pca, scaler



# ===== Part 7: Mutual Information 防御方法 =====
def mutual_info_clustering_selection(args, epoch):
    MODELS_PATH = args.get_save_model_folder_path()
    model_files = sorted(os.listdir(MODELS_PATH))

    start_file = get_model_files_for_suffix(get_model_files_for_epoch(model_files, epoch), args.get_epoch_save_start_suffix())[0]
    start_model = load_models(args, [os.path.join(MODELS_PATH, start_file)])[0]
    start_vec = flatten_layers(start_model.get_nn_parameters())

    end_files = get_model_files_for_suffix(get_model_files_for_epoch(model_files, epoch), args.get_epoch_save_end_suffix())
    mi_scores, worker_ids = [], []

    for f in end_files:
        worker_id = get_worker_num_from_model_file_name(f)
        end_model = load_models(args, [os.path.join(MODELS_PATH, f)])[0]
        end_vec = flatten_layers(end_model.get_nn_parameters())
        score = mutual_info_regression(start_vec.reshape(-1, 1), end_vec)[0]
        mi_scores.append(score)
        worker_ids.append(worker_id)

    sorted_mi = sorted(zip(mi_scores, worker_ids), key=lambda x: x[0], reverse=True)
    fed_pct  = getattr(args, "fed_pct", 0.2)
    grey_pct = getattr(args, "grey_pct", 0.5)
    logger.info(f"[MI@E{epoch}] fed_pct={fed_pct}, grey_pct={grey_pct}")
    #下面三句是添加小样本兜底
    fed_count  = max(1, int(len(sorted_mi) * fed_pct))
    grey_count = max(1, int(len(sorted_mi) * grey_pct))
    grey_count = min(grey_count, max(0, len(sorted_mi) - fed_count))


    fed_idx = [x[1] for x in sorted_mi[:fed_count]]
    benign_idx = [x[1] for x in sorted_mi[:-grey_count]]
    grey_idx = [x[1] for x in sorted_mi[-grey_count:]]
    bad_idx = []

    return benign_idx, bad_idx, grey_idx, fed_idx

# ===== Part 8: 联邦保留验证方法 =====
def verify_by_fr(clients, fr_idx, grey_idx):
    results = []
    for g in grey_idx:
        votes = 0
        for f in fr_idx:
            acc = clients[f].validate(clients[g].get_nn_parameters())
            if acc >= clients[f].test_acc * 0.95:
                votes += 1
        results.append(votes >= len(fr_idx) / 2)
    return results

def class_validation(clients, fr_idx, grey_idx):
    cls_pass, acc_pass = [], []
    for g in grey_idx:
        cls_votes, acc_votes = 0, 0
        for f in fr_idx:
            diff, all_cls = clients[f].by_class_validate(clients[g].get_nn_parameters())
            if diff <= clients[f].class_diff:
                cls_votes += 1
            if all_cls:
                acc_votes += 1
        cls_pass.append(cls_votes >= len(fr_idx) / 2)
        acc_pass.append(acc_votes >= len(fr_idx) / 2)
    return cls_pass, acc_pass
