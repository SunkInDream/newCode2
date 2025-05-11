import os
import torch
import torch.nn.functional as F
from models_TCDF import ADDSTCN
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def process_single_matrix(args):
    idx, x_np, mask_np, causal_matrix, model_params, epochs, lr, gpu_id = args
    
    # 关键修改1：明确设置GPU环境变量
    if gpu_id != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda:0'  # 在当前进程中只能看到一个GPU，所以始终是cuda:0
    else:
        device = 'cpu'
        
    # 声明使用指定设备
    print(f"处理矩阵 {idx}，使用设备: {device}")
    
    seq_len, total_features = x_np.shape
    filled = x_np.copy()

    for target in range(total_features):
        causal_indices = list(np.where(causal_matrix[:, target] == 1)[0])
        if target not in causal_indices:
            causal_indices.append(target)
        else:
            causal_indices.remove(target)
            causal_indices.append(target)
        causal_indices = causal_indices[:3] + [target]  # 限制为3+1列

        input_data = x_np[:, causal_indices].T[np.newaxis, ...]  # (1, 4, seq_len)
        target_data = x_np[:, target][np.newaxis, :, np.newaxis]  # (1, seq_len, 1)
        target_mask = mask_np[:, target][np.newaxis, :, np.newaxis]

        x = torch.tensor(input_data, dtype=torch.float32).to(device)
        y = torch.tensor(target_data, dtype=torch.float32).to(device)
        mask = torch.tensor(target_mask, dtype=torch.float32).to(device)

        model = ADDSTCN(target, input_size=4, cuda=(device=='cuda:0'), **model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 关键修改2：添加进度报告
        for epoch in range(epochs):
            model.train()
            pred = model(x)
            loss = F.mse_loss(pred * mask, y * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 每5轮显示进度
            if epoch % 5 == 0:
                print(f"矩阵 {idx}, 特征 {target}/{total_features}, Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()
            missing_idx = np.where(mask_np[:, target] == 0)[0]
            filled[missing_idx, target] = pred[missing_idx]

    return idx, filled

def train_all_features_parallel(dataset, model_params, epochs=10, lr=0.001):
    assert dataset.total_causal_matrix is not None, "total_causal_matrix is required"
    causal_matrix = dataset.total_causal_matrix
    
    # 获取实际可用的GPU ID
    available_gpus = list(range(torch.cuda.device_count())) or ['cpu']
    print(f"可用GPU: {available_gpus}")
    
    # 准备任务列表，每个任务包含所有必要参数
    tasks = [
        (i, dataset.initial_filled[i], dataset.mask_data[i], causal_matrix, 
         model_params, epochs, lr, available_gpus[i % len(available_gpus)])
        for i in range(len(dataset.initial_filled))
    ]
    
    # 使用与可用GPU数量相同的工作进程
    with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
        for idx, filled_matrix in executor.map(process_single_matrix, tasks):
            dataset.final_filled[idx] = filled_matrix
            print(f"矩阵 {idx} 处理完成")