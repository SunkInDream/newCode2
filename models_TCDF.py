import copy 
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from models_TCN import ADDSTCN
import torch.nn.functional as F

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
    torch.cuda.set_device(device) if device != 'cpu' else None

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
    x, mask, columns = prepare_data(file_or_array)
    num_features = x.shape[1]

    device = gpu_id if torch.cuda.device_count() > 0 else 'cpu'

    # 串行处理各个特征
    results = []
    for i in range(num_features):
        results.append(run_single_task((i, file_or_array, params, device)))

    matrix = np.zeros((num_features, num_features), dtype=int)
    for tgt, causes in results:
        for c in causes:
            matrix[tgt, c] = 1
    return matrix, columns

