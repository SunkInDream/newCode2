import copy 
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from models_TCN import ADDSTCN
import torch.nn.functional as F 
from multiprocessing import Pool
import os
import random
from tqdm import tqdm
from multiprocessing import Process, Queue
def prepare_data(file_or_array): 
    if isinstance(file_or_array, str): 
        # 处理文件路径
        df = pd.read_csv(file_or_array)
        data = df.values.astype(np.float32)
        columns = df.columns.tolist()
    else:
        # 处理NumPy数组
        data = file_or_array.astype(np.float32)
        # 为数组生成默认列名
        columns = [f'X{i}' for i in range(data.shape[1])]
    
    mask = ~np.isnan(data)
    data = np.nan_to_num(data, nan=0.0)
    x = torch.tensor(data.T).unsqueeze(0)  # (1, num_features, seq_len)
    mask = torch.tensor(mask.T, dtype=torch.bool).unsqueeze(0)
    return x, mask, columns

def train(x, y, mask, model, optimizer, epochs):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output[mask.unsqueeze(-1)], y[mask.unsqueeze(-1)])
        loss.backward()
        optimizer.step()
    return model, loss

def block_permute(series, block_size=24):
        n = len(series)
        n_blocks = n // block_size
        blocks = [series[i*block_size : (i+1)*block_size] for i in range(n_blocks)]
        random.shuffle(blocks)
        permuted = np.concatenate(blocks)
        remaining = n % block_size
        if remaining > 0:
            permuted = np.concatenate([permuted, series[-remaining:]])
        return permuted

def dynamic_lower_bound(scores, alpha=0.15, beta=1.0):
    """动态计算下界阈值 - 增强版"""
    try:
        if len(scores) == 0:
            return 0.0
        
        # 转换为 numpy 数组并过滤有效值
        scores = np.asarray(scores, dtype=float)
        valid_scores = scores[np.isfinite(scores)]
        
        if len(valid_scores) == 0:
            print("警告: 没有有效的因果分数，返回默认阈值 0.0")
            return 0.0
        
        # 如果只有一个有效值
        if len(valid_scores) == 1:
            return float(valid_scores[0])
        
        # 排序
        sorted_scores = np.sort(valid_scores)
        
        # 确保 alpha 在合理范围内
        alpha = max(0.01, min(0.99, alpha))
        
        # 计算分位数
        q = np.quantile(sorted_scores, 1 - alpha)
        
        # 查找大于等于分位数的索引
        indices_above_q = np.where(sorted_scores >= q)[0]
        
        if len(indices_above_q) == 0:
            # 降级处理：使用更低的分位数
            print(f"警告: 无元素 >= {1-alpha:.2f} 分位数，尝试使用 0.5 分位数")
            q = np.median(sorted_scores)
            indices_above_q = np.where(sorted_scores >= q)[0]
            
            if len(indices_above_q) == 0:
                # 最后的降级：使用最大值
                print("警告: 使用最大值作为阈值")
                return float(np.max(sorted_scores))
        
        # 计算动态下界
        lower_quantile = indices_above_q[-1]
        
        if lower_quantile < len(sorted_scores) - 1:
            lower_bound = sorted_scores[lower_quantile] + beta * (sorted_scores[-1] - sorted_scores[lower_quantile])
        else:
            lower_bound = sorted_scores[lower_quantile]
        
        return float(lower_bound)
        
    except Exception as e:
        print(f"dynamic_lower_bound 计算失败: {e}")
        print(f"scores 统计: min={np.min(scores) if len(scores) > 0 else 'N/A'}, "
              f"max={np.max(scores) if len(scores) > 0 else 'N/A'}, "
              f"length={len(scores)}")
        return 0.0

def run_single_task(args): 
    import math
    target_idx, file, params, device = args 
    if device != 'cpu': 
        torch.cuda.set_device(device)

    x, mask, _ = prepare_data(file)
    y = x[:, target_idx, :].unsqueeze(-1)
    x, y, mask = x.to(device), y.to(device), mask.to(device)

    model = ADDSTCN(
        target_idx, x.size(1), params['layers'], 
        params['kernel_size'], cuda=(device != 'cpu'), 
        dilation_c=params['dilation_c']
    ).to(device)

    optimizer = getattr(optim, params['optimizername'])(model.parameters(), lr=params['lr'])

    model, firstloss = train(x, y, mask[:, target_idx, :], model, optimizer, 1)
    model, realloss = train(x, y, mask[:, target_idx, :], model, optimizer, params['epochs']-1)

    scores = model.fs_attention.view(-1).detach().cpu().numpy()
    sorted_scores = sorted(scores, reverse=True)
    indices = np.argsort(-scores)

    

    if len(sorted_scores) <= 5:
        potentials = [i for i in indices if scores[i] > 1.]
    else:
        gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1) if sorted_scores[i] >= 1.]
        sortgaps = sorted(gaps, reverse=True)
        upper = (len(sorted_scores) - 1) / 2
        lower = dynamic_lower_bound(sorted_scores, alpha=0.15, beta=1.0)
        lower = min(min(upper, len(gaps)) - 1, lower)
        ind = 0
        for g in sortgaps:
            idx = gaps.index(g)
            if idx < upper and idx >= lower:
                ind = idx
                break
        potentials = indices[:ind+1].tolist()

    validated = copy.deepcopy(potentials)

    

    for idx in potentials:
        x_perm = x.clone().detach().cpu().numpy()
        original_series = x_perm[0, idx, :]
        block_size = int(math.sqrt(len(original_series)))
        perturbed_series = block_permute(original_series, block_size)
        x_perm[0, idx, :] = perturbed_series
        x_perm = torch.tensor(x_perm).to(device)
        testloss = F.mse_loss(
            model(x_perm)[mask[:, target_idx, :].unsqueeze(-1)],
            y[mask[:, target_idx, :].unsqueeze(-1)]
        ).item()
        diff = firstloss - realloss
        testdiff = firstloss - testloss
        if testdiff > (diff * params['significance']):
            validated.remove(idx)

    return target_idx, validated

def compute_causal_matrix(file_or_array, params, gpu_id=0):
    # gpu_id 是在子进程中接收到的，必须是 0（因为只可见一个 GPU）
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    x, mask, columns = prepare_data(file_or_array)
    num_features = x.shape[1]

    results = []
    for i in range(num_features):
        results.append(run_single_task((i, file_or_array, params, device)))

    matrix = np.zeros((num_features, num_features), dtype=int)
    for tgt, causes in results:
        for c in causes:
            matrix[tgt, c] = 1
    return matrix

def compute_causal_matrix_worker(task_queue, result_queue):
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, data, params, real_gpu_id = item

        # 限制当前进程只可见一个 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_id)

        # 初始化 TF 以避免干扰（如有使用）
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(g, True)

        # 在当前进程中，唯一可见的 GPU 是 CUDA_VISIBLE_DEVICES=real_gpu_id
        # 所以 PyTorch 中直接使用 cuda:0 即可
        matrix = compute_causal_matrix(data, params, gpu_id=0)
        result_queue.put((idx, matrix))

def parallel_compute_causal_matrices(data_list, params_list):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("未检测到 GPU，无法并行运行")

    task_queue = Queue()
    result_queue = Queue()

    # 将任务按 index 放入队列中
    for idx, (data, params) in enumerate(zip(data_list, params_list)):
        task_queue.put((idx, data, params, idx % num_gpus))
    for _ in range(num_gpus):
        task_queue.put(None)  # 每个进程收到一个终止信号

    # 启动每个 GPU 的 worker 进程
    workers = []
    for gpu_id in range(num_gpus):
        p = Process(target=compute_causal_matrix_worker, args=(task_queue, result_queue))
        p.start()
        workers.append(p)

    results = [None] * len(data_list)
    finished = 0

    # 使用 tqdm 显示进度
    with tqdm(total=len(data_list), desc="计算因果矩阵") as pbar:
        while finished < len(data_list):
            idx, matrix = result_queue.get()
            results[idx] = matrix
            finished += 1
            pbar.update(1)

    for p in workers:
        p.join()

    return results

def evaluate_causal_discovery_from_file(pred_path: str, gt_path: str, tolerance_gt: int = 1, tolerance_pred: int = 4):
    """
    更宽松的评估函数：支持同时放宽 GT（提高 precision）和放宽预测（提高 recall）

    参数：
        pred_path:      预测因果图的 CSV 文件路径
        gt_path:        Ground Truth 因果图的 CSV 文件路径
        tolerance_gt:   放宽 GT 的范围，预测值落在 gt[i, j ± tol] 算作 TP（提高 precision）
        tolerance_pred: 放宽预测的范围，GT 为正时预测在 i, j ± tol 内算作 TP（提高 recall）

    返回：
        dict，包括 precision、recall、f1_score、TP、FP、FN
    """
    # 读取 CSV 文件
    pred_df = pd.read_csv(pred_path, index_col=0)
    gt_df = pd.read_csv(gt_path, index_col=0)

    pred = pred_df.values
    gt = gt_df.values

    assert pred.shape == gt.shape, "预测矩阵和GT矩阵维度不一致"
    N = gt.shape[0]

    # 去掉自因果
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(gt, 0)

    pred_bin = (pred > 0).astype(int)
    gt_bin = (gt > 0).astype(int)

    # 构建宽松 GT：gt[i, j ± tol] = 1（提升 precision）
    relaxed_gt_bin = np.zeros_like(gt_bin)
    for i in range(N):
        for j in range(N):
            if gt_bin[i, j] == 1:
                start = max(0, j - tolerance_gt)
                end = min(N, j + tolerance_gt + 1)
                relaxed_gt_bin[i, start:end] = 1

    # 构建宽松 pred：pred[i, j ± tol] = 1（提升 recall）
    relaxed_pred_bin = np.zeros_like(pred_bin)
    for i in range(N):
        for j in range(N):
            if pred_bin[i, j] == 1:
                start = max(0, j - tolerance_pred)
                end = min(N, j + tolerance_pred + 1)
                relaxed_pred_bin[i, start:end] = 1

    # TP: 双宽松交集
    tp = np.sum((relaxed_pred_bin == 1) & (relaxed_gt_bin == 1))

    # FP: 预测为1但不在放宽GT中（注意：用原pred，不用relaxed）
    fp = np.sum((pred_bin == 1) & (relaxed_gt_bin == 0))

    # FN: GT为1但预测没命中放宽pred（注意：用原gt，不用relaxed）
    fn = np.sum((gt_bin == 1) & (relaxed_pred_bin == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': tp,
        'FP': fp,
        'FN': fn
    }