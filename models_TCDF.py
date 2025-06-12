import copy 
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from models_TCN import ADDSTCN
import torch.nn.functional as F 
from multiprocessing import Pool
import os
from tqdm import tqdm
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

def run_single_task(args):
    target_idx, file, params, device = args
    if device != 'cpu':
        torch.cuda.set_device(device)
    
    x, mask, _ = prepare_data(file)
    y = x[:, target_idx, :].unsqueeze(-1)
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    model = ADDSTCN(target_idx, x.size(1), params['layers'], params['kernel_size'], cuda=(device != 'cpu'), dilation_c=params['dilation_c']).to(device)
    optimizer = getattr(optim, params['optimizername'])(model.parameters(), lr=params['lr'])
    model, firstloss = train(x, y, mask[:, target_idx, :], model, optimizer, 1)
    model, realloss = train(x, y, mask[:, target_idx, :], model, optimizer, params['epochs']-1)
    scores = model.fs_attention.view(-1).detach().cpu().numpy()
    sorted_scores = sorted(scores, reverse=True)
    indices = np.argsort(-scores)
    potentials = []
    if len(sorted_scores) <= 5:
        potentials = [i for i in indices if scores[i] > 1.]
    else:
        gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1) if sorted_scores[i] >= 1.]
        sortgaps = sorted(gaps, reverse=True)
        ind = 0
        for g in sortgaps:
            idx = gaps.index(g)
            if idx < (len(sorted_scores) - 1) / 2 and idx > 0:
                ind = idx
                break
        potentials = indices[:ind+1].tolist()
    validated = copy.deepcopy(potentials)
    for idx in potentials:
        x_perm = x.clone().detach().cpu().numpy()
        np.random.shuffle(x_perm[0, idx, :])
        x_perm = torch.tensor(x_perm).to(device)
        testloss = F.mse_loss(
            model(x_perm)[mask[:, target_idx, :].unsqueeze(-1)],
            y[mask[:, target_idx, :].unsqueeze(-1)]
        ).item()
        diff = firstloss-realloss
        testdiff = firstloss-testloss
        if testdiff>(diff * params['significance']):
             validated.remove(idx) 
    return target_idx, validated

def compute_causal_matrix_with_gpu(args):
    """包装函数，用于多进程调用"""
    file_or_array, params, gpu_id = args
    return compute_causal_matrix(file_or_array, params, gpu_id)

def compute_causal_matrix(file_or_array, params, gpu_id=0):
    x, mask, columns = prepare_data(file_or_array)
    num_features = x.shape[1]
    device = f'cuda:{gpu_id}' if torch.cuda.device_count() > gpu_id else 'cpu'
    
    # 串行处理各个特征
    results = [] 
    for i in range(num_features): 
        results.append(run_single_task((i, file_or_array, params, device))) 
 
    matrix = np.zeros((num_features, num_features), dtype=int) 
    for tgt, causes in results: 
        for c in causes: 
            matrix[tgt, c] = 1 
    return matrix

def parallel_compute_causal_matrices(data_list, params_list, max_workers=None):
    """
    并行计算多个因果矩阵
    
    Args:
        data_list: 数据文件路径或numpy数组的列表
        params_list: 对应的参数列表
        max_workers: 最大工作进程数，默认为GPU数量
    
    Returns:
        results: 因果矩阵结果列表
    """
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("警告: 未检测到GPU，将使用CPU运行")
        max_workers = max_workers or os.cpu_count()
        gpu_ids = ['cpu'] * len(data_list)
    else:
        max_workers = max_workers or num_gpus
        # 循环分配GPU
        gpu_ids = [i % num_gpus for i in range(len(data_list))]
    
    print(f"使用 {max_workers} 个进程并行处理 {len(data_list)} 个矩阵")
    
    # 准备参数
    args_list = [(data, params, gpu_id) 
                 for data, params, gpu_id in zip(data_list, params_list, gpu_ids)]
    
    # 并行执行
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap(compute_causal_matrix_with_gpu, args_list),
            total=len(args_list),
            desc="计算因果矩阵",
            ncols=80
        ))
    
    return results