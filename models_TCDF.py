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