import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from models_CAUSAL import *

class MyDataset(Dataset):
    def __init__(self, file_paths):   
        self.original_data = []        # 含nan的原始数据 (list of ndarray)
        self.mask_data = []            # 掩码矩阵，1表示有值，0表示缺失 (list of ndarray)
        self.initial_filled = []       # 初次填补的结果 (list of ndarray)
        self.final_filled = []         # 最终填补结果 (list of ndarray, 默认为None)
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
            second_prepro_data = SecondProcess(file)
            second_prepro_data = second_prepro_data.values.astype(np.float32)
            self.initial_filled.append(second_prepro_data)
            self.final_filled.append(None)
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return {
            'original': torch.tensor(self.original_data[idx]),
            'mask': torch.tensor(self.mask_data[idx]),
            'initial_filled': torch.tensor(self.initial_filled[idx]),
            'final_filled': torch.tensor(self.final_filled[idx]) if self.final_filled[idx] is not None else None,
            #'label': self.labels[idx],
            #'total_causal_matrix': torch.tensor(self.total_causal_matrix) if self.total_causal_matrix is not None else None
        }

