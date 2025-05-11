import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from models_CAUSAL import *
from models_TCDF import ADDSTCN
def compute_single_causal(args):
    mat, params, gpu = args
    return compute_causal_matrix(mat, params, gpu)[0]
def run_single_sample(args):
    idx, matrix, mask, causal_matrix, params, device = args
    torch.cuda.set_device(device)
    matrix = torch.tensor(matrix, dtype=torch.float32).to(device)
    mask = torch.tensor(mask, dtype=torch.bool).to(device)
    causal_matrix = torch.tensor(causal_matrix, dtype=torch.int).to(device)

    time_steps, num_features = matrix.shape
    x_list, y_list, m_list = [], [], []

    for target in range(num_features):
        # 获取前三个因果关系
        causes = torch.where(causal_matrix[:, target] == 1)[0][:3].tolist()
        if not causes:  # 如果没有因果关系，使用自身
            causes = [target]
        selected = causes + [target] if target not in causes else causes
        
        x = matrix[:, selected].T.unsqueeze(0)  # [1, len(selected), time_steps]
        y = matrix[:, target].unsqueeze(0).unsqueeze(0)  # [1, 1, time_steps]
        m = mask[:, target].unsqueeze(0).unsqueeze(0)  # [1, 1, time_steps]
        x_list.append(x)
        y_list.append(y)
        m_list.append(m)

    x_batch = torch.cat(x_list, dim=0)  # [num_features, len(selected), time_steps]
    y_batch = torch.cat(y_list, dim=0)  # [num_features, 1, time_steps]
    m_batch = torch.cat(m_list, dim=0)  # [num_features, 1, time_steps]

    model = ADDSTCN(0, 4, params['layers'], params['kernel_size'], cuda=True, dilation_c=params['dilation_c']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    for _ in range(params['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(x_batch)  # [num_features, time_steps, 1]
        out_reshaped = out.permute(0, 2, 1)  # [num_features, 1, time_steps]
        loss = torch.nn.functional.mse_loss(
            out_reshaped[m_batch].view(-1), 
            y_batch[m_batch].view(-1)
        )
        loss.backward()
        optimizer.step()

    model.eval()
    result = matrix.cpu().numpy().copy()  # 创建副本很重要
    
    with torch.no_grad():
        pred = model(x_batch)  # [num_features, time_steps, 1]
        pred_reshaped = pred.permute(0, 2, 1)  # [num_features, 1, time_steps]
        
        # 转换为NumPy并处理缺失值填充
        pred_np = pred_reshaped.cpu().numpy()  # [num_features, 1, time_steps]
        mask_np = m_batch.cpu().numpy()  # [num_features, 1, time_steps]
        
        # 按特征循环填充
        for i in range(num_features):
            # 找出当前特征的缺失位置（布尔数组形式）
            missing_mask = ~mask_np[i, 0]  # [time_steps]
            missing_indices = np.where(missing_mask)[0]  # 一维索引数组
            
            # 填充预测值
            if len(missing_indices) > 0:
                result[missing_indices, i] = pred_np[i, 0, missing_indices]

    return idx, result
class MyDataset(Dataset):
    def __init__(self, file_paths): 
        self.file_paths = file_paths  
        self.original_data = []        # 含nan的原始数据 (list of ndarray)
        self.mask_data = []            # 掩码矩阵，1表示有值，0表示缺失 (list of ndarray)
        self.initial_filled = []       # 初次填补的结果 (list of ndarray)
        self.final_filled = []         # 最终填补结果 (list of ndarray, 默认为None)
        self.center_repre = []       # 中心表示 (list of ndarray)
        self.labels = []          # 标签 (list)
        self.total_causal_matrix = None  # 总因果矩阵 (ndarray, 默认为None)
        files = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
        for file in files:
            df = pd.read_csv(file)
            data = df.values.astype(np.float32)
            self.original_data.append(data)
            first_prepro_data =  FirstProcess(file)
            mask = (~first_prepro_data.isna()).values.astype(np.float32)
            self.mask_data.append(mask)
            second_prepro_data = SecondProcess(first_prepro_data)
            second_prepro_data = second_prepro_data.values.astype(np.float32)
            self.initial_filled.append(second_prepro_data)
            self.final_filled.append(None)
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return {
            'original': self.original_data[idx],
            'mask': self.mask_data[idx],
            'initial_filled': self.initial_filled[idx],
            'final_filled':self.final_filled[idx]
            #'label': self.labels[idx],
            #'total_causal_matrix': torch.tensor(self.total_causal_matrix) if self.total_causal_matrix is not None else None
        }
    def causal_dis(self, n_cluster, params=None):
        # 仅执行聚类，返回中心表示
        data = np.array([np.nanmean(x, axis=0) for x in self.initial_filled])
        data = np.nan_to_num(data, nan=0.0)
        km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
        labels = km.fit_predict(data)

        self.labels = labels  # 保存标签
        self.center_repre = []
        
        for k in range(n_cluster):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0:  # 处理空聚类
                continue
            cluster_data = data[idxs]
            dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
            best_idx = idxs[np.argmin(dists)]
            self.center_repre.append(self.initial_filled[best_idx])
            
        return self.center_repre  # 返回中心表示，不进行因果发现    
    def impute_with_tcn(self, params):
        gpus = list(range(torch.cuda.device_count())) or ['cpu']
        tasks = [
            (i, self.initial_filled[i], self.mask_data[i], self.total_causal_matrix, params, gpus[i % len(gpus)])
            for i in range(len(self.initial_filled))
        ]

        with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
            for idx, filled in executor.map(run_single_sample, tasks):
                self.final_filled[idx] = filled
